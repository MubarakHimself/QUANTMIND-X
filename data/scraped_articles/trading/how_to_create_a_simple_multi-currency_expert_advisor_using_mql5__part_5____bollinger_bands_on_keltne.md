---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5):  Bollinger Bands On Keltner Channel — Indicators Signal
url: https://www.mql5.com/en/articles/13861
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 16
scraped_at: 2026-01-22T17:08:47.985533
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/13861&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048822326964887383)

MetaTrader 5 / Trading


### Introduction

The definition of a multi-currency Expert Advisor in this article is an Expert Advisor or trading robot that can trade (open orders, close orders and manage orders, for example: Trailing Stop Loss and Trailing Profit) for more than 1 symbol pair from only one symbol chart, where in this article Expert Advisor will trade for 30 pairs.

In this article we will use signals from two indicators, in this case Bollinger Bands® On Keltner Channel.

On the old platform (MetaTrader 4) the use of signals like this was known as the iBandsOnArray function.

In the explanation of the iBandsOnArray function in the MQL4 Reference, it is stated that: Note: Unlike iBands(...), the iBandsOnArray() function does not take data by symbol name, timeframe, the applied price. The price data must be previously prepared...

In the discussion on the MQL5 forum, I have read that there are even traders or users who say that iBandOnArray() does not exist in MQL5.

Indeed, iBandOnArray() is not in the MQL5 function list, but by using the iBands() indicator handle, we can actually easily create iBandOnArray() in MQL5. In fact, in my opinion, using the indicator handle in MQL5 is easier and more convenient than using the iBandsOnArray() function in MetaTrader 4.

As proven in previous [articles](https://www.mql5.com/en/articles/13770), we all know that multi-currency trading, both on the trading terminal and on the strategy tester, is possible with the power, capabilities and facilities provided by MQL5.

Therefore, the goal is to satisfy the essential needs of traders who want efficient and effective trading robots, so by relying on the strengths, capabilities and facilities provided by the highly reliable MQL5, we can create a simple Multi-Currency Expert Advisor, which in this article uses 2 indicator signals for open orders: Bollinger Bands® On Keltner Channel, where the Bollinger Bands indicator will use price data from the Keltner Channel indicator. Meanwhile, for trailing stops we will still use the Parabolic SAR (iSAR) indicator.

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is designed to trade on a symbol or pair as follows:

EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP,EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD,GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,NZDJPY, CADCHF, CADJPY, CHFJPY = 28 pairs

Plus 2 Metal pairs: XAUUSD (Gold) and XAGUSD (Silver)

Total is 30 pairs.

As in the previous article, in this article we make it easy by adding special input properties for the pair name prefix and pair name suffix. Then, with a simple function, we will handle the pair name prefix and/or suffix in combination with the 30 registered pair names, so that when using an EA on MetaTrader 5 from a broker with such special symbol names, everything will work smoothly.

The weakness of the function to detect symbol names with prefixes and suffixes is that this function only works on Forex and Metal symbol pairs or names in MT5, but not on special symbols and indices. Apart from that, another weakness of this method is if the trader makes a typo (must be case sensitive) for the name of the pair prefix and/or suffix. Therefore, for traders who will use the Expert Advisor in this article, we expect precision and accuracy in typing the pair name prefix and/or suffix.

As in the previous article, in this Expert Advisor we have also added 10 options for the pairs that will be traded at this time. One of the 10 option pairs that will be traded is "Trader's Desired Pairs", where the pairs to be traded must be manually entered by the trader in the Expert Input property. But you must always keep in mind that the name of the pair you are entering must already be in the list of the 30 pairs.

In this version of the Expert Advisor we have also added an option for trading session (time zone), so the pairs that will trade can correspond to the time for the trading session.

**2\. Signal indicators.**

In this version of the Expert Advisor we are going to use the signals of two indicators, in this case the Bollinger Bands® on the Keltner Channel. As price data for the Bollinger Bands® indicator, we will use the Keltner Channel indicator.

2.1. Keltner Channel Indicator.

The Keltner Channel was first introduced by Chester Keltner in the 1960s. The original formula used simple moving averages (SMAs) and the high/low price range to calculate the bands.  In the 1980s, a new formula was introduced that used the Average True Range (ATR). The ATR method is commonly used today.

The Keltner Channel indicator used for Expert Advisor s in this article, I specifically created using a method that is popular today, namely the Exponential Moving Average (EMA) period 20 with upper and lower bands using the ATR indicator period 20.

Keltner Channel indicator input properties are as follows:

![BBOnKC_Keltner Channel Indicator-cr](https://c.mql5.com/2/61/BBOnKC_Keltner_Channel_Indicator-cr.png)

2.2. Bollinger Bands® Indicator.

The Bollinger Bands® were created by John Bollinger in the 80's and quickly became one of the most widely used indicators in the technical analysis field. Bollinger Bands® consist of three bands - the upper, middle and lower bands - that are used to highlight short-term price extremes in the market. The upper band is a sign of overbought conditions, while the lower band is a sign of oversold conditions. Most financial analysts will use Bollinger Bands® and combine them with other indicators to get a better analytical picture of the state of the market.

In the Expert Advisor of this article we will use the Bollinger Bands® indicator with period 38, which uses price data from the Keltner Channel indicator.

Bollinger Bands® indicator input properties are as follows:

![BBOnKC_Bollinger Bands Indicator-cr](https://c.mql5.com/2/61/BBOnKC_Bollinger_Bands_Indicator-cr.png)

An illustration of the Keltner Channel indicator as price data for the Bollinger Bands® indicator for BUY or SELL SELL signals can be seen in Figures 1 and 2.

![BBOnKC_BUY signal](https://c.mql5.com/2/61/BBOnKC_BUY_signal.png)

Figure 1. Buy Signal

![BBOnKC_SELL signal](https://c.mql5.com/2/61/BBOnKC_SELL_signal.png)

Figure 2. Sell Signal

In the illustration above, a signal is given only when the middle line of the Keltner channel crosses above or below the upper line of the Bollinger Bands® or the lower line of the Bollinger Bands®. But for the Expert Advisor in this article, the indicator signal is actually a crossover between the middle line of the Keltner Channel indicator and the upper, middle and lower lines of the Bollinger Bands® indicator.

- For Buy Signals:

1. First signal: When the middle line of the Keltner Channel indicator crosses up the lower line of the Bollinger Bands® indicator; or
2. Second signal: When the middle line of the Keltner Channel indicator crosses up the middle line of the Bollinger Bands® indicator; or
3. Third signal: When the middle line of the Keltner Channel indicator crosses up the upper line of the Bollinger Bands® indicator.

- For Sell Signals:

1. First signal: When the middle line of the Keltner Channel indicator crosses down the upper line of the Bollinger Bands® indicator; or
2. Second signal: When the middle line of the Keltner Channel indicator crosses down the middle line of the Bollinger Bands® indicator; or
3. Third signal: When the middle line of the Keltner Channel indicator crosses down the lower line of the Bollinger Bands® indicator.

**3\. Trade & Order Management**

This multi-currency Expert Advisor gives you several options for managing your trades:

3.1. Stop Loss Orders

Options: Use Order Stop Loss (Yes) or (No)

- If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.

- If the option Use Order Stop Loss (Yes): Again given the option: Use Automatic Calculation Stop Loss (Yes) or (No)

- If the option Automatic Calculation Stop Loss (Yes),  then the Stop Loss calculation will be performed automatically by the Expert.

- If the option Automatic Calculation Stop Loss (No), then the trader must Input Stop Loss value in Pips.

- If the option Use Order Stop Loss (No): Then the Expert will check for each order opened, whether the signal condition is still good and order may be maintained in a profit or condition the signal has weakened and the order needs to be closed to save profit or signal condition has reversed direction and order must be closed in a loss condition.

> Note: Especially for Close Trade and Save profit due to weak signal, an option is given, whether to activate it or not.

If it is not activated (No), even though the signal has weakened, the order will still be maintained or will not be closed to save profit.

If activated (Yes), the conditions for the Keltner Channer and Bollinger Bands® indicators are:

- For close the Buy orders: When the lower line of the Keltner Channel indicator crosses down the upper line of Bollinger Bands®, the Buy order will be closed.
- For close the Sell orders: When the upper line of the Keltner Channel indicator crosses up the lower line of Bollinger Bands®, the Sell order will be closed.

The code for setting a stop loss order is as follows:

```
double MCEA::OrderSLSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice)
  {
//---
    slv=0.0;
    int x=PairsIdxArray(xsymb);
    Pips(xsymb);
    RefreshTick(xsymb);
    //--
    switch(type)
      {
       case (ORDER_TYPE_BUY):
         {
           if(use_sl==Yes && autosl==Yes) slv=mc_symbol.NormalizePrice(atprice-38*pip);
           else
           if(use_sl==Yes && autosl==No)  slv=mc_symbol.NormalizePrice(atprice-SLval*pip);
           else slv=0.0;
           //--
           break;
         }
       case (ORDER_TYPE_SELL):
         {
           if(use_sl==Yes && autosl==Yes) slv=mc_symbol.NormalizePrice(atprice+38*pip);
           else
           if(use_sl==Yes && autosl==No)  slv=mc_symbol.NormalizePrice(atprice+SLval*pip);
           else slv=0.0;
         }
      }
    //---
    return(slv);
//---
  } //-end OrderSLSet()
//---------//
```

Code for Close Trade and Save profit due to weak signal is as follows:

```
int MCEA::GetCloseInWeakSignal(const string symbol,int exis) // Signal Indicator Position Close in profit
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int br=3;
    Pips(symbol);
    double difud=mc_symbol.NormalizePrice(1.5*pip);
    //--
    double KCub[],
           KClb[];
    double BBub[],
           BBlb[];
    //--
    ArrayResize(KCub,br,br);
    ArrayResize(KClb,br,br);
    ArrayResize(BBub,br,br);
    ArrayResize(BBlb,br,br);
    ArraySetAsSeries(KCub,true);
    ArraySetAsSeries(KClb,true);
    ArraySetAsSeries(BBub,true);
    ArraySetAsSeries(BBlb,true);
    //--
    int xx=PairsIdxArray(symbol);
    //--
    CopyBuffer(hKC[xx],1,0,br,KCub);
    CopyBuffer(hKC[xx],2,0,br,KClb);
    CopyBuffer(hBB[xx],1,0,br,BBub);
    CopyBuffer(hBB[xx],2,0,br,BBlb);
    //--
    int dirmove=DirectionMove(symbol,TFt);
    bool closebuy=(KClb[1]>=BBub[1] && KClb[0]<BBub[0]-difud);
    bool closesel=(KCub[1]<=BBlb[1] && KCub[0]>BBlb[0]+difud);
    //--
    if(exis==down && closesel && dirmove==rise) ret=rise;
    if(exis==rise && closebuy && dirmove==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetCloseInWeakSignal()
//---------//
```

3.2. Take Profit orders

Options: Use Order Take Profit (Yes) or (No)

- If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

- If the option Use Order Take Profit (Yes): Again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)

- If the option Automatic Calculation Order Take Profit (Yes), then the calculation of the Take Profit Order will be carried out automatically by the Expert.

- If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

The code for setting a take profit order is as follows:

```
double MCEA::OrderTPSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice)
  {
//---
    tpv=0.0;
    int x=PairsIdxArray(xsymb);
    Pips(xsymb);
    RefreshTick(xsymb);
    //--
    switch(type)
      {
       case (ORDER_TYPE_BUY):
         {
           if(use_tp==Yes && autotp==Yes) tpv=mc_symbol.NormalizePrice(atprice+50*pip);
           else
           if(use_tp==Yes && autotp==No)  tpv=mc_symbol.NormalizePrice(atprice+TPval*pip);
           else tpv=0.0;
           //--
           break;
         }
       case (ORDER_TYPE_SELL):
         {
           if(use_tp==Yes && autotp==Yes) tpv=mc_symbol.NormalizePrice(atprice-50*pip);
           else
           if(use_tp==Yes && autotp==No)  tpv=mc_symbol.NormalizePrice(atprice-TPval*pip);
           else tpv=0.0;
         }
      }
    //---
    return(tpv);
//---
  } //-end OrderTPSet()
//---------//
```

3.3. Trailing Stop and Trailing Take Profit

Options: Use Trailing SL/TP (Yes) or (No)

- If the Use Trailing SL/TP option is (No), then the Expert will not do trailing stop loss and trailing take profit.

- If the option Use Trailing SL/TP (Yes): Again given the option: Use Automatic Trailing (Yes) or (No)

- If the option Use Automatic Trailing (Yes), then the trailing stop will be executed by the Expert using the Parabolic SAR (iSAR) value on the same with timeframe signal calculations, and at the same time by making trailing profit based on the variable value TPmin (minimum trailing profit value).
- If the option Use Automatic Trailing (No), then the trailing stop will be performed by the Expert using the value in the input property.

> Note: The Expert will carry out a trailing take profit simultaneously with a trailing stop.

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
            double PSAR[];
            ArrayResize(PSAR,br,br);
            ArraySetAsSeries(PSAR,true);
            CopyBuffer(hParIU[x],0,0,br,PSAR);
            RefreshPrice(xsymb,TFt,br);
            //--
            if(ptype==POSITION_TYPE_BUY  && (PSAR[0]<iLow(xsymb,TFt,0)))
               pval=PSAR[0];
            if(ptype==POSITION_TYPE_SELL && (PSAR[0]>iHigh(xsymb,TFt,0)))
               pval=PSAR[0];
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

> 4.1. Set SL / TP All Orders

> When the trader input parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No),

> but then the trader intends to use stop loss or take profit on all orders, then with just a single click of the button

> "Set SL / TP All Orders" all orders will be modified and a stop loss will be applied and/or take profits.

> 4.2. Close All Orders If a trader wants to close all orders, then with just a single click of the button "Close All Orders" all open orders will be closed.

> 4.3. Close All Orders Profit If a trader wants to close all orders that are already profitable, then with a single click of the button "Close All Orders Profit" all open orders that are already profitable will be closed.

**5\. Management Orders and Chart Symbols.**

For multi-currency Expert Advisors who will trade 30 pairs from only one chart symbol, it will be very effective and efficient to provide a button panel for all symbols so that traders can change charts or symbols with just one click to see the accuracy of the indicator signal when the expert opens or closes an order.

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
   Australia,          // Trading on Australia Sydney Session
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

Enumeration to use Money Management Lot size

```
//--
enum YN
  {
   No,
   Yes
  };
//--
```

Enumeration to select the timeframe that will be used in calculation signal indicators

```
//--
enum TFUSE
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

Note: With TFUSE enumeration, we limit the use of time frame calculations for experts only starting from TF-M5 to TF-D1

Expert input properties

```
//---
input group               "=== Global Strategy EA Parameter ==="; // Global Strategy EA Parameter
input TFUSE               tfinuse = TFH1;             // Select Expert TimeFrame, default PERIOD_H1
input int                KCPeriod = 20;               // Input Keltner Channel Period, default 20
input ENUM_MA_METHOD     KCMethod = MODE_EMA;         // Select Keltner Channel MA Method, default EMA
input ENUM_APPLIED_PRICE   KCMAAP = PRICE_TYPICAL;    // Select Keltner Channel MA Applied Price, default Price Typical
input int                KCATRPer = 20;               // Input Keltner Channel ATR Period, default 20
input double      KCATRBandsMulti = 1.0;              // Input Keltner Channel ATR bands multiplier
input int                BBPeriod = 38;               // Input Bollinger Bands® Indicator period, default 38
input double               BBDevi = 1.0;              // Input Bollinger Bands® Indicator Deviations, default 1.00
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
input YN                  use_tp = Yes;              // Use Order Take Profit (Yes) or (No)
input YN                  autotp = Yes;              // Use Automatic Calculation Take Profit (Yes) or (No)
input double               TPval = 100;              // If Not Use Automatic TP - Input TP value in Pips
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
input ulong              magicEA = 20231204;         // Expert ID (Magic Number)
//---
```

Note: If the input property of Expert ID (Magic Number) is left blank, the Expert Advisor will be able to manage orders opened manually.

In the Global Strategy EA Parameters expert input property group, traders are instructed to select the Expert Timeframe for indicator signal calculations and to enter parameters for the Keltner Channel and Bollinger Bands® indicators.

When creating the Expert Advisor in this article, the input properties for the Keltner Channel indicator in the Expert Advisor will use exactly the same input properties as those in the indicator.

```
//--
input int                period_kc = 20;             // Input Keltner Channel Period
input ENUM_MA_METHOD     ma_method = MODE_EMA;       // Select MA Type of smoothing
input ENUM_APPLIED_PRICE  ma_price = PRICE_TYPICAL;  // Select MA Applied Price
input int               atr_period = 20;             // Input ATR Period (typically over 10 or 20)
input double            band_multi = 1.00;           // Input the Band Multiplier ATR Desired
//--
```

In the expert input property group Select Pairs to Trade, the traders must choose the pair to trade from the 10 options provided, by default All Forex 30 Pairs is determined.

To configure the pair to be traded, we will call the HandlingSymbolArrays() function. With HandlingSymbolArrays() function we will handle all pairs that will be traded.

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
        Alert("-- "+expname+" -- ",Symbol()," -- Expert Advisor will be Remove from the chart.");
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

Inside the HandlingSymbolArrays() function we will call the SetSymbolNamePS() function. Using SetSymbolNamePS() we will be able to handle symbol names that are prefixed and/or suffixed.

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

Note: The expert will validate the pairs. If the trader makes a mistake in the entry of the pair name or the pair prefix name and/or the pair suffix name (typos), or In case of failure of pair validation, the expert will have a warning and the Expert Advisor will be removed from the chart.

In the expert input property group Trade on Specific Time, here the trader will choose to Trade on Specific Time Zone (Yes) or (No) and If Yes, select the enumeration options:

- Trading on Custom Session
- Trading on New Zealand Session
- Trading on Australia Sydney Session
- Trading on Asia Tokyo Session
- Trading on Europe London Session
- Trading on America New York Session

Trading on Custom Session: In this session, traders will need to set the time or the hours and minutes to start trading and the hours and minutes to close trading. This means that the EA will only carry out activities during the specified time period from the start to the end.

In New Zealand Session Trading to New York US Session Trading, the time from the start of the trade to the close of the trade is calculated by the EA.

Class for working Expert Advisor

To determine the construction and configuration in the Expert Advisor workflow, we create a class that declares all the variables, objects and functions required by this Multi-Currency Expert Advisor.

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
   double            SARstep,
                     SARmaxi;
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
    int              BBOnKeltnerChannel(const string symbol);
    int              PARSAR05(const string symbol);
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

    //-- BBOnKeltnerChannel_MCEA Config --
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
    int              hKC[];
    int              hBB[];
    int              hParIU[],
                     hPar05[];
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
                     TFT05;
    //--
    bool             PanelExtra;
    //------------
                     MCEA(void);
                     ~MCEA(void);
    //------------
    //--
    virtual void     BBOnKeltnerChannel_MCEA_Config(void);
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
    int              GetOpenPosition(const string symbol);
    int              GetCloseInWeakSignal(const string symbol,int exis);
    //--
    string           getUninitReasonText(int reasonCode);
    //--
    //------------
//---
  }; //-end class MCEA
//---------//
```

The very first, most important function in the Multi-Currency Expert Advisor work process that is called by OnInit() is BBOnKeltnerChannel\_MCEA\_Config().

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//---
   mc.BBOnKeltnerChannel_MCEA_Config();
   //--
   return(INIT_SUCCEEDED);
//---
  } //-end OnInit()
//---------//
```

The BBOnKeltnerChannel\_MCEA\_Config() function configures all symbols to be used, all handle indicators to be used, and some important functions of the include file header for the Expert Advisor workflow.

On the function line 468 to 484, it is explained how to handle timeframe and create indicator handles for all indicators that will be used.

```
//+------------------------------------------------------------------+
//| Expert Configuration                                             |
//+------------------------------------------------------------------+
void MCEA::BBOnKeltnerChannel_MCEA_Config(void)
  {
//---
    //--
    HandlingSymbolArrays(); // With this function we will handle all pairs that will be traded
    //--
    TFT05=PERIOD_M5;
    ENUM_TIMEFRAMES TFs[]={PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};
    int arTFs=ArraySize(TFs);
    //--
    for(int x=0; x<arTFs; x++) if(tfinuse==x) TFt=TFs[x]; // TF for calculation signal
    //--
    //-- Keltner Channel and Bollinger Bands® Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hKC[x]=iCustom(DIRI[x],TFt,indiname,KCPeriod,KCMethod,KCMAAP,KCATRPer,KCATRBandsMulti); //-- Handle for the Keltner Channel indicator
        hParIU[x]=iSAR(DIRI[x],TFt,SARstep,SARmaxi);         //-- Handle for the iSAR indicator according to the selected Timeframe
        hPar05[x]=iSAR(DIRI[x],TFT05,SARstep,SARmaxi);       //-- Handle for the iSAR indicator for M5 Timeframe
        //--
      }
    //--
    for(int x=0; x<arrsymbx; x++)
      hBB[x]=iBands(DIRI[x],TFt,BBPeriod,0,BBDevi,hKC[x]); //-- Handle for the iBands On Keltner Channel Indicator handle
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
  } //-end BBOnKeltnerChannel_MCEA_Config()
//---------//
```

**2\. Expert tick function**

Within the Expert Tick function (OnTick() function) we will call one of the most important functions in a multi-currency Expert Advisor, namely ExpertActionTrade() function.

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

This is the sequence of the EA work process within this function.

The ExpertActionTrade() function will perform all activities and manage automatic trading, starting from opening orders, closing orders, trailing stops or trading profits and other additional activities.

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

The Day Trading On/Off property group gives traders the option to trade on the specified days from Sunday to Saturday. In this way, traders can activate or deactivate the experts for trading on the specified day by using the (Yes) or (No) option.

```
//--Day Trading On/Off
input group               "=== Day Trading On/Off ==="; // Day Trading On/Off
input YN                    ttd0 = No;               // Select Trading on Sunday (Yes) or (No)
input YN                    ttd1 = Yes;              // Select Trading on Monday (Yes) or (No)
input YN                    ttd2 = Yes;              // Select Trading on Tuesday (Yes) or (No)
input YN                    ttd3 = Yes;              // Select Trading on Wednesday (Yes) or (No)
input YN                    ttd4 = Yes;              // Select Trading on Thursday (Yes) or (No)
input YN                    ttd5 = Yes;              // Select Trading on Friday (Yes) or (No)
input YN                    ttd6 = No;               // Select Trading on Saturday (Yes) or (No)
```

Execution for Day Trading On/Off is as follows:

```
bool MCEA::TradingToday(void)
  {
//---
    bool tradetoday=false;
    int trdday=ThisTime(dow);
    hariini="No";
    //--
    int ttd[];
    ArrayResize(ttd,7);
    ttd[0]=ttd0;
    ttd[1]=ttd1;
    ttd[2]=ttd2;
    ttd[3]=ttd3;
    ttd[4]=ttd4;
    ttd[5]=ttd5;
    ttd[6]=ttd6;
    //--
    if(ttd[trdday]==Yes) {tradetoday=true; hariini="Yes";}
   //--
   return(tradetoday);
//---
  } //-end TradingToday()
//---------//
```

Notes: Day Trading On/Off conditions will be displayed in the Trading Info on Chart.

In the Expert Advisor's "Trade on Specific Time" property group, traders have the option to trade by time zone.

```
input group               "=== Trade on Specific Time ==="; // Trade on Specific Time
input YN           trd_time_zone = Yes;              // Select If You Like to Trade on Specific Time Zone
input tm_zone            session = Cus_Session;      // Select Trading Time Zone
input swhour            stsescuh = hr_00;            // Time Hour to Start Trading Custom Session (0-23)
input inmnt             stsescum = mn_15;            // Time Minute to Start Trading Custom Session (0-55)
input swhour            clsescuh = hr_23;            // Time Hour to Stop Trading Custom Session (0-23)
input inmnt             clsescum = mn_55;            // Time Minute to Stop Trading Custom Session (0-55)
```

Notes: As explained above, in the case of trading on New Zealand Session to trading on US New York Session, the time from the start of trading to the close of trading is calculated by the EA. Thus, in the Expert Entry properties, traders only need to set the time for the hour and minute when trading starts and the time for the hour and minute when trading ends for the Custom Session.

Specifically for time zone trading, a call to the boolean Trade\_session() function is added to the ExpertActionTrade() function.

If Trade\_session() is true, then the EA work process will continue until it is finished, but if it is false, then the EA will only perform the tasks "Close Trade and Save Profit due to Weak Signal if it is (Yes)" and "Trailing Stop if it is (Yes)".

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

To get a signal, the ExpertActionTrade() function calls the GetOpenPosition() function.

```
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int BBOnKC=BBOnKeltnerChannel(symbol);
    //--
    if(BBOnKC==rise) ret=rise;
    if(BBOnKC==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetOpenPosition()
//---------//
```

The GetOpenPosition() function calls BBOnKeltnerChannel() functions that perform signal calculations.

```
int MCEA::BBOnKeltnerChannel(const string symbol) // Fungction of Bollinger Bands® On Keltner Channel Indicator
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int br=3;
    Pips(symbol);
    double difud=mc_symbol.NormalizePrice(1.5*pip);
    //--
    double KCmb[], // Destination array for Keltner Channel Middle Line buffer
           KCub[], // Destination array for Keltner Channel Upper Band buffer
           KClb[]; // Destination array for Keltner Channel Lower Band buffer
    double BBmb[], // Destination array for Bollinger Bands® BASE_LINE buffer
           BBub[], // Destination array for Bollinger Bands® UPPER_BAND buffer
           BBlb[]; // Destination array for Bollinger Bands® LOWER_BAND buffer
    //--
    ArrayResize(KCmb,br,br);
    ArrayResize(KCub,br,br);
    ArrayResize(KClb,br,br);
    ArrayResize(BBmb,br,br);
    ArrayResize(BBub,br,br);
    ArrayResize(BBlb,br,br);
    ArraySetAsSeries(KCmb,true);
    ArraySetAsSeries(KCub,true);
    ArraySetAsSeries(KClb,true);
    ArraySetAsSeries(BBmb,true);
    ArraySetAsSeries(BBub,true);
    ArraySetAsSeries(BBlb,true);
    //--
    int xx=PairsIdxArray(symbol);
    //--
    CopyBuffer(hKC[xx],0,0,br,KCmb);
    CopyBuffer(hKC[xx],1,0,br,KCub);
    CopyBuffer(hKC[xx],2,0,br,KClb);
    CopyBuffer(hBB[xx],0,0,br,BBmb);
    CopyBuffer(hBB[xx],1,0,br,BBub);
    CopyBuffer(hBB[xx],2,0,br,BBlb);
    //--
    bool BBKCrise1=(KCmb[1]<=BBlb[1] && KCmb[0]>BBlb[0]+difud);
    bool BBKCrise2=(KCmb[1]<=BBmb[1] && KCmb[0]>BBmb[0]+difud);
    bool BBKCrise3=(KCmb[1]<=BBub[1] && KCmb[0]>BBub[0]+difud);
    //--
    bool BBKCdown1=(KCmb[1]>=BBub[1] && KCmb[0]<BBub[0]-difud);
    bool BBKCdown2=(KCmb[1]>=BBmb[1] && KCmb[0]<BBmb[0]-difud);
    bool BBKCdown3=(KCmb[1]>=BBlb[1] && KCmb[0]<BBlb[0]-difud);
    //--
    if(BBKCrise1 || BBKCrise2 || BBKCrise3) ret=rise;
    if(BBKCdown1 || BBKCdown2 || BBKCdown3) ret=down;
    //--
    return(ret);
//---
  } //-end BBOnKeltnerChannel()
//---------//
```

As you can see, inside the BBOnKeltnerChannel() function, we use and call 1 function, which is the PairsIdxArray() function.

```
int xx=PairsIdxArray(symbol);
```

```
int MCEA::PairsIdxArray(const string symbol)
  {
//---
    int pidx=-1;
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
//---------//
```

The PairsIdxArray() function is used to get the name of the requested symbol and the handles of its indicators. Then the corresponding indicator handle is called to get the buffer value of the Keltner Channel and Bollinger Bands® signal from that timeframe.

Next, we have to know the buffer number of the Keltner Channel indicator. In the Keltner Channel indicator, we know in the OnInit() function, that  the buffer numbers are the following: 0 - Middle Line, 1 - Upper Band, 2 - Lower Band

```
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   //-- assignment of array to indicator buffer
   SetIndexBuffer(0,KC_Middle,INDICATOR_DATA);
   SetIndexBuffer(1,KC_UpperB,INDICATOR_DATA);
   SetIndexBuffer(2,KC_LowerB,INDICATOR_DATA);
   SetIndexBuffer(3,KC_ATRtemp,INDICATOR_CALCULATIONS);
   //--
   handleMA=iMA(Symbol(),Period(),period_kc,0,MODE_EMA,ma_price);
   //--
   if(handleMA==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the Moving Average indicator for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }
   //--
   handleATR=iATR(Symbol(),Period(),atr_period);
   //--
   if(handleATR==INVALID_HANDLE)
     {
      //--- tell about the failure and output the error code
      PrintFormat("Failed to create handle of the ATR indicator for the symbol %s/%s, error code %d",
                  Symbol(),
                  EnumToString(Period()),
                  GetLastError());
      //--- the indicator is stopped early
      return(INIT_FAILED);
     }
   //--
   short_name=StringFormat("Keltner Channel(%d, %s, %.2f)",period_kc,EnumToString(ma_method),band_multi);
   IndicatorSetString(INDICATOR_SHORTNAME,short_name);
   IndicatorSetInteger(INDICATOR_DIGITS,Digits());
   //--
   return(INIT_SUCCEEDED);
//---
  }
//---------//
```

In this case, we will also need to know the number of buffers of the Bollinger Bands® indicator.

Meanwhile, as explained in the [iBands®](https://www.mql5.com/en/docs/indicators/ibands) function that:

"Note The buffer numbers are the following: 0 - BASE\_LINE, 1 - UPPER\_BAND, 2 - LOWER\_BAND"

applied\_price

> \[in\] The price used. Can be any of the price constants ENUM\_APPLIED\_PRICE or a handle of another indicator.

In this article, Expert Advisors will use the applied price for Bollinger Bands® from the Keltner Channel indicator handle.

So, to get the buffer value for each line of the Keltner Channel and Bollinger Bands® indicators, we will copy each buffer from the indicators handle.

To copy the Middle Line buffer (buffer 0) from the Keltner Channel indicator handle to the destination array:

```
CopyBuffer(hKC[xx],0,0,br,KCmb);
```

To copy the Upper Band buffer (buffer 1) from the Keltner Channel indicator handle to the destination array:

```
CopyBuffer(hKC[xx],1,0,br,KCub);
```

To copy the Lower Band buffer (buffer 2) from the Keltner Channel indicator handle to the destination array:

```
CopyBuffer(hKC[xx],2,0,br,KClb);
```

To copy the BASE\_LINE buffer (buffer 0) from the Bollinger Bands® indicator handle to the destination array:

```
CopyBuffer(hBB[xx],0,0,br,BBmb);
```

To copy the UPPER\_BAND buffer (buffer 1) from the Bollinger Bands® indicator handle to the destination array:

```
CopyBuffer(hBB[xx],1,0,br,BBub);
```

To copy the LOWER\_BAND buffer (buffer 2) from the Bollinger Bands® indicator handle to the destination array:

```
CopyBuffer(hBB[xx],2,0,br,BBlb);
```

Next, the following results are returned by the GetOpenPosition() function:

- Value 0, signal unknown.
- Value 1, is a signal for open Buy order.
- Value -1, is a signal for open Sell order.

The Expert Advisor calls the OpenBuy() function when the GetOpenPosition() function returns the value 1.

```
bool MCEA::OpenBuy(const string symbol)
  {
//---
    ResetLastError();
    //--
    bool buyopen      = false;
    string ldComm     = GetCommentForOrder()+"_Buy";
    double ldLot      = MLots(symbol);
    ENUM_ORDER_TYPE type_req = ORDER_TYPE_BUY;
    //--
    MqlTradeRequest req={};
    MqlTradeResult  res={};
    MqlTradeCheckResult check={};
    //-- structure is set to zero
    ZeroMemory(req);
    ZeroMemory(res);
    ZeroMemory(check);
    //--
    CurrentSymbolSet(symbol);
    double SL=OrderSLSet(symbol,type_req,mc_symbol.Bid());
    double TP=OrderTPSet(symbol,type_req,mc_symbol.Ask());
    //--
    if(RefreshTick(symbol))
       buyopen=mc_trade.Buy(ldLot,symbol,mc_symbol.Ask(),SL,TP,ldComm);
    //--
    int error=GetLastError();
    if(buyopen||error==0)
      {
        string bsopen="Open BUY Order for "+symbol+" ~ Ticket= ["+(string)mc_trade.ResultOrder()+"] successfully..!";
        Do_Alerts(symbol,bsopen);
      }
    else
      {
        mc_trade.CheckResult(check);
        Do_Alerts(Symbol(),"Open BUY order for "+symbol+" FAILED!!. Return code= "+
                 (string)mc_trade.ResultRetcode()+". Code description: ["+mc_trade.ResultRetcodeDescription()+"]");
        return(false);
      }
    //--
    return(buyopen);
    //--
//---
  } //-end OpenBuy
//---------//
```

In the meantime, the Expert Advisor will call the OpenSell() function if the GetOpenPosition() function returns a value of -1.

```
bool MCEA::OpenSell(const string symbol)
  {
//---
    ResetLastError();
    //--
    bool selopen      = false;
    string sdComm     = GetCommentForOrder()+"_Sell";
    double sdLot      = MLots(symbol);
    ENUM_ORDER_TYPE type_req = ORDER_TYPE_SELL;
    //--
    MqlTradeRequest req={};
    MqlTradeResult  res={};
    MqlTradeCheckResult check={};
    //-- structure is set to zero
    ZeroMemory(req);
    ZeroMemory(res);
    ZeroMemory(check);
    //--
    CurrentSymbolSet(symbol);
    double SL=OrderSLSet(symbol,type_req,mc_symbol.Ask());
    double TP=OrderTPSet(symbol,type_req,mc_symbol.Bid());
    //--
    if(RefreshTick(symbol))
       selopen=mc_trade.Sell(sdLot,symbol,mc_symbol.Bid(),SL,TP,sdComm);
    //--
    int error=GetLastError();
    if(selopen||error==0)
      {
        string bsopen="Open SELL Order for "+symbol+" ~ Ticket= ["+(string)mc_trade.ResultOrder()+"] successfully..!";
        Do_Alerts(symbol,bsopen);
      }
    else
      {
        mc_trade.CheckResult(check);
        Do_Alerts(Symbol(),"Open SELL order for "+symbol+" FAILED!!. Return code= "+
                 (string)mc_trade.ResultRetcode()+". Code description: ["+mc_trade.ResultRetcodeDescription()+"]");
        return(false);
      }
    //--
    return(selopen);
    //--
//---
  } //-end OpenSell
//---------//
```

**4\. ChartEvent Function.**

To support effectiveness and efficiency in using Multi-Currency Expert Advisors, it is considered necessary to create one or more manual buttons in managing orders and changing charts or symbols.

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
           Alert("-- "+mc.expname+" -- ",Symbol()," -- Expert Advisor will be Remove from the chart.");
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

In the Other Expert Advisor Parameters group of input properties, the trader will be given the option to select whether to display trading information on the chart (Yes) or (No).

If this option is selected (Yes), trade information will be displayed on the chart to which the Expert Advisor is attached by calling the TradeInfo() function.

As part of the TradeInfo() function, we have also added a function to describe the time according to trading time zone conditions.

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

The interface of the Multi-Currency Expert Advisor BBOnKeltnerChannel\_MCEA looks similar to the following figure.

![EA-looks](https://c.mql5.com/2/61/EA-looks.png)

Under the Expert Advisor name BBOnKeltnerChannel\_MCEA as you can see there are buttons "M", "C" and "R"

When the M button is clicked, a panel of buttons for manual clicks will be displayed, as shown in the figure below.

![Expert_manual_button_01](https://c.mql5.com/2/61/Expert_manual_button_01.png)

The trader can manage orders manually when the manual click button panel is displayed:

1\. Set SL/TP All Orders As explained above, if the trader enters the parameters Use Order Stop Loss (No) and/or Use Order Take Profit (No), but then the trader intends to use Stop Loss or Take Profit on all orders, then a single click on the "Set SL / TP All Orders" button will modify all orders and apply Stop Loss and/or Take Profit.

```
void MCEA::SetSLTPOrders(void)
  {
//---
   ResetLastError();
   MqlTradeRequest req={};
   MqlTradeResult  res={};
   MqlTradeCheckResult check={};
   //--
   double modbuysl=0;
   double modselsl=0;
   double modbuytp=0;
   double modseltp=0;
   string position_symbol;
   int totalorder=PositionsTotal();
   //--
   for(int i=totalorder-1; i>=0; i--)
     {
       string symbol=PositionGetSymbol(i);
       position_symbol=symbol;
       if(mc_position.Magic()==magicEA)
         {
           ENUM_POSITION_TYPE opstype = mc_position.PositionType();
           if(opstype==POSITION_TYPE_BUY)
             {
               Pips(symbol);
               RefreshTick(symbol);
               double price    = mc_position.PriceCurrent();
               double pos_open = mc_position.PriceOpen();
               double pos_stop = mc_position.StopLoss();
               double pos_take = mc_position.TakeProfit();
               modbuysl=SetOrderSL(symbol,opstype,pos_open);
               if(price<modbuysl) modbuysl=mc_symbol.NormalizePrice(price-slip*pip);
               modbuytp=SetOrderTP(symbol,opstype,pos_open);
               if(price>modbuytp) modbuytp=mc_symbol.NormalizePrice(price+slip*pip);
               //--
               if(pos_stop==0.0 || pos_take==0.0)
                 {
                   if(!mc_trade.PositionModify(position_symbol,modbuysl,modbuytp))
                     {
                       mc_trade.CheckResult(check);
                       Do_Alerts(symbol,"Set SL and TP for "+EnumToString(opstype)+" on "+symbol+" FAILED!!. Return code= "+
                                (string)mc_trade.ResultRetcode()+". Code description: ["+mc_trade.ResultRetcodeDescription()+"]");
                     }
                 }
             }
           if(opstype==POSITION_TYPE_SELL)
             {
               Pips(symbol);
               RefreshTick(symbol);
               double price    = mc_position.PriceCurrent();
               double pos_open = mc_position.PriceOpen();
               double pos_stop = mc_position.StopLoss();
               double pos_take = mc_position.TakeProfit();
               modselsl=SetOrderSL(symbol,opstype,pos_open);
               if(price>modselsl) modselsl=mc_symbol.NormalizePrice(price+slip*pip);
               modseltp=SetOrderTP(symbol,opstype,pos_open);
               if(price<modseltp) modseltp=mc_symbol.NormalizePrice(price-slip*pip);
               //--
               if(pos_stop==0.0 || pos_take==0.0)
                 {
                   if(!mc_trade.PositionModify(position_symbol,modselsl,modseltp))
                     {
                       mc_trade.CheckResult(check);
                       Do_Alerts(symbol,"Set SL and TP for "+EnumToString(opstype)+" on "+symbol+" FAILED!!. Return code= "+
                                (string)mc_trade.ResultRetcode()+". Code description: ["+mc_trade.ResultRetcodeDescription()+"]");
                     }
                 }
             }
         }
     }
    //--
    return;
//---
  } //-end SetSLTPOrders
//---------//
```

2\. Close All Orders If a trader wants to close all the orders, then with a single click on the "Close All Orders" button, all the open orders will be closed.

```
void MCEA::CloseAllOrders(void) //-- function: close all order
   {
//----
    ResetLastError();
    //--
    MqlTradeRequest req={};
    MqlTradeResult  res={};
    MqlTradeCheckResult check={};
    //--
    int total=PositionsTotal(); // number of open positions
    //--- iterate over all open positions
    for(int i=total-1; i>=0; i--)
      {
        //--- if the MagicNumber matches
        if(mc_position.Magic()==magicEA)
          {
            //--
            string position_Symbol   = PositionGetSymbol(i);  // symbol of the position
            ulong  position_ticket   = PositionGetTicket(i);  // ticket of the the opposite position
            ENUM_POSITION_TYPE  type = mc_position.PositionType();
            RefreshTick(position_Symbol);
            bool closepos = mc_trade.PositionClose(position_Symbol,slip);
            //--- output information about the closure
            PrintFormat("Close #%I64d %s %s",position_ticket,position_Symbol,EnumToString(type));
            //---
          }
      }
   //---
   return;
//----
   } //-end CloseAllOrders()
//---------//
```

3\. Close All Profits If a trader wants to close all orders that are already profitable, a single click on the "Close All Profitable Orders" button will close all open orders that are already profitable.

```
bool MCEA::ManualCloseAllProfit(void)
   {
//----
    ResetLastError();
    //--
    bool orclose=false;
    //--
    MqlTradeRequest req={};
    MqlTradeResult  res={};
    MqlTradeCheckResult check={};
    //--
    int ttlorder=PositionsTotal(); // number of open positions
    //--
    for(int x=0; x<arrsymbx; x++)
       {
         string symbol=DIRI[x];
         orclose=false;
         //--
         for(int i=ttlorder-1; i>=0; i--)
            {
              string position_Symbol   = PositionGetSymbol(i);
              ENUM_POSITION_TYPE  type = mc_position.PositionType();
              if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
                {
                  double pos_profit = mc_position.Profit();
                  double pos_swap   = mc_position.Swap();
                  double pos_comm   = mc_position.Commission();
                  double cur_profit = NormalizeDouble(pos_profit+pos_swap+pos_comm,2);
                  ulong  position_ticket = PositionGetTicket(i);
                  //---
                  if(type==POSITION_TYPE_BUY && cur_profit>0.02)
                    {
                      RefreshTick(position_Symbol);
                      orclose = mc_trade.PositionClose(position_Symbol,slip);
                      //--- output information about the closure
                      PrintFormat("Close #%I64d %s %s",position_ticket,position_Symbol,EnumToString(type));
                    }
                  if(type==POSITION_TYPE_SELL && cur_profit>0.02)
                    {
                      RefreshTick(position_Symbol);
                      orclose = mc_trade.PositionClose(position_Symbol,slip);
                      //--- output information about the closure
                      PrintFormat("Close #%I64d %s %s",position_ticket,position_Symbol,EnumToString(type));
                    }
                }
            }
       }
     //--
     return(orclose);
//----
   } //-end ManualCloseAllProfit()
//---------//
```

When the C button is clicked, a panel button with 30 symbol names or pairs is displayed and traders can click on one of the pair names or symbol names. Clicking on one of the pair names or symbols immediately replaces the chart symbol with the symbol whose name was clicked.

![Expert_manual_button_02](https://c.mql5.com/2/61/Expert_manual_button_02.png)

In this case, the OnChartEvent() function is called ChangeChartSymbol() function when one of the symbol names is clicked.

```
       //--- if Symbol button is click
       if(lensparam==lensymbol)
         {
           int sx=mc.ValidatePairs(sparam);
           ChangeChartSymbol(mc.AS30[sx],CCS);
           mc.PanelExtra=false;
         }
       //--
```

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

Clicking on the R button will remove the Multi-Currency Expert Advisor BBOnKeltnerChannel\_MCEA from the chart, so traders don't have to disconnect the experts manually.

```
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
       //--
       //--- if "R" button is click
       if(sparam=="R")
         {
           Alert("-- "+mc.expname+" -- ",Symbol()," -- Expert Advisor will be Remove from the chart.");
           ExpertRemove();
           //--- unpress the button
           ObjectSetInteger(0,"R",OBJPROP_STATE,false);
           ObjectSetInteger(0,"R",OBJPROP_ZORDER,0);
           if(!ChartSetSymbolPeriod(0,Symbol(),Period()))
             ChartSetSymbolPeriod(0,Symbol(),Period());
           DeletePanelButton();
           ChartRedraw(0);
         }
       //---
     }
```

### Strategy Tester

As you know, the MetaTrader 5 terminal Strategy Tester supports and allows us to [test strategies](https://www.mql5.com/en/docs/runtime/testing), trade on multiple symbols or test automatic trading for all available symbols and on all available timeframes. Therefore, on the MetaTrader 5 Strategy Tester platform we will be testing a BBOnKeltnerChannel\_MCEA as a Multi-Currency Expert Advisor.

In this test, we have placed the BBOnKeltnerChannel\_MCEA on the XAGUSD pair and the H1 time frame, with a custom time period of 2023.09.04 to 2023.12.02.

![ST_tester-period](https://c.mql5.com/2/62/ST_tester-period.png)

The test was carried out with two different input properties, specifically in the Trade & Order Management parameters group.

1\. Default Input Properties.

![ST_Default-Input1](https://c.mql5.com/2/62/ST_Default-Input1.png)

![ST_Default-Input-result](https://c.mql5.com/2/62/ST_Default-Input-result.png)

2\. Custom Input Properties.

![ST_Custom-Input1](https://c.mql5.com/2/62/ST_Custom-Input1.png)

![ST_Custom-Input-result](https://c.mql5.com/2/62/ST_Custom-Input-result.png)

### Conclusion

The conclusion in creating a Multi-Currency Expert Advisor with signals from the Bollinger Bands® indicator on the Keltner Channel indicator using MQL5 is as follows:

1. It turns out that creating a multi-currency Expert Advisor in MQL5 is very simple and not much different from creating a single-currency Expert Advisor.
2. Compared to the old platform (MetaTrader 4), using the indicator handle in MQL5 to get indicator values and signals is easier and more convenient.
3. Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders by eliminating the need to open many chart symbols for trading.
4. Applying the right trading strategy will increase the probability of profit compared to using a single currency Expert Advisor. This is because losses in one pair will be covered by profits in other pairs.
5. This BBOnKeltnerChannel\_MCEA Multi-Currency Expert Advisor is just a sample for learning and idea generation. The test results on the Strategy Tester are still not good. Therefore, by experimenting and testing on different time frames or different indicator period calculations, it is possible to get better and more profitable results.
6. The test results on the Strategy Tester of BBOnKeltnerChannel\_MCEA show that the results of custom input parameter properties are better than the default input parameter properties.

We hope that this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders to learn and develop ideas.

Thanks for reading.

Note: If you have an idea for creating a simple Multi-Currency Expert Advisor based on built-in MQL5 standard indicator signals, please suggest it in the comments.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13861.zip "Download all attachments in the single ZIP archive")

[BBOnKeltnerChannel\_MCEA.mq5](https://www.mql5.com/en/articles/download/13861/bbonkeltnerchannel_mcea.mq5 "Download BBOnKeltnerChannel_MCEA.mq5")(214.31 KB)

[Keltner\_Channel.mq5](https://www.mql5.com/en/articles/download/13861/keltner_channel.mq5 "Download Keltner_Channel.mq5")(8.01 KB)

[BBOnKeltnerChannel.tpl](https://www.mql5.com/en/articles/download/13861/bbonkeltnerchannel.tpl "Download BBOnKeltnerChannel.tpl")(5.01 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR](https://www.mql5.com/en/articles/13008)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/458880)**
(13)


![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
10 Jan 2024 at 06:19

**hdhyxiaobin [#](https://www.mql5.com/en/forum/458880#comment_51595778):**

Hello roberto, I have loaded the two indicators required by the EA onto the chart, but the EA still prompts that these two indicators cannot be opened.

see attached

Thanks

automated translation applied by moderator

As can be seen in the image you provided, the Keltner Channel indicator file name you used is wrong.

Please see the difference in indicator file names.

[![6a1p_20240109203550_ed](https://c.mql5.com/3/426/6a1p_20240109203550_ed__1.jpg)](https://c.mql5.com/3/426/6a1p_20240109203550_ed.jpg "https://c.mql5.com/3/426/6a1p_20240109203550_ed.jpg")

![wrong_file_name](https://c.mql5.com/3/426/wrong_file_name.png)

You are using different indicators than those required by expert advisors.

![hdhyxiaobin](https://c.mql5.com/avatar/avatar_na2.png)

**[hdhyxiaobin](https://www.mql5.com/en/users/hdhyxiaobin)**
\|
13 Jan 2024 at 03:32

**Roberto Jacobs [#](https://www.mql5.com/en/forum/458880#comment_51612236):**

As can be seen in the image you provided, the Keltner Channel indicator file name you used is wrong.

Please see the difference in indicator file names.

You are using different indicators than those required by expert advisors.

hello roberto

have modify indicator name and EA running   OK

but open many orders at one symbol

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
13 Jan 2024 at 11:41

**hdhyxiaobin [#](https://www.mql5.com/en/forum/458880#comment_51687660):**

hello roberto

have modify indicator name and EA running   OK

but open many orders at one symbol

Maybe you don't understand or don't know that on the internet there are many versions of the Keltner Channel indicator.

Did you read my explanation in the article, or did you just download the EA program.

In part 2. Signal Indicators, I have explained:

"The Keltner Channel indicator used for Expert Advisors in this article, I specifically created using a method that is popular today, namely the Exponential Moving Average (EMA) period 20 with upper and lower bands using the ATR indicator period 20."

So, don't hope that by renaming the indicator the EA will work correctly, because different indicators will have different input parameters and different calculations.

If you don't use the Keltner Channel indicator that I created specifically for the Bollinger Bands® On Keltner Channel EA, you should not expect to be able to use the Bollinger Bands® On Keltner Channel EA.

![navtrack1959](https://c.mql5.com/avatar/avatar_na2.png)

**[navtrack1959](https://www.mql5.com/en/users/navtrack1959)**
\|
22 Feb 2024 at 19:29

Hello Roberto,

You have done a great job ! I hope to make it work on my Pc.

I do have the following message : Keltner Channel.ex5 open error and others to follow.

Of cause, no result on the tester.

Thank you for your help.

Alain

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
23 Feb 2024 at 12:30

**navtrack1959 [#](https://www.mql5.com/en/forum/458880/page2#comment_52373056):**

Hello Roberto,

You have done a great job ! I hope to make it work on my Pc.

I do have the following message : Keltner Channel.ex5 open error and others to follow.

Of cause, no result on the tester.

Thank you for your help.

Alain

Hello,

My Expert Advisor name is BBOnKeltnerChannel\_MCEA, NOT Multi-currency BB-Keltner (in the picture you sent me in email the name of EA is Multi-currency BB-Keltner)

The name of the indicator that I created specifically for BBOnKeltnerChannel\_MCEA is Keltner Channel.mq5 (view and download the attachment to this article)

There are many versions of the Keltner Channel indicator on the web.

Please pay close attention to the file name and property input in indicators and experts because this is case sensitive.

If the input properties are not the same then the program will not run.

If you modify the EA program and an error occurs, you should not ask me.

All programs in MQL5 have been validated by the moderator in charge of it, so if they do not pass validation it is impossible for a program to be published by MQL5.

![Data Science and Machine Learning (Part 16): A Refreshing Look at Decision Trees](https://c.mql5.com/2/62/1midjourney_image_13862_46_406__3_logo.png)[Data Science and Machine Learning (Part 16): A Refreshing Look at Decision Trees](https://www.mql5.com/en/articles/13862)

Dive into the intricate world of decision trees in the latest installment of our Data Science and Machine Learning series. Tailored for traders seeking strategic insights, this article serves as a comprehensive recap, shedding light on the powerful role decision trees play in the analysis of market trends. Explore the roots and branches of these algorithmic trees, unlocking their potential to enhance your trading decisions. Join us for a refreshing perspective on decision trees and discover how they can be your allies in navigating the complexities of financial markets.

![Neural networks made easy (Part 54): Using random encoder for efficient research (RE3)](https://c.mql5.com/2/57/random_encoder_for_efficient_exploration_054_avatar.png)[Neural networks made easy (Part 54): Using random encoder for efficient research (RE3)](https://www.mql5.com/en/articles/13158)

Whenever we consider reinforcement learning methods, we are faced with the issue of efficiently exploring the environment. Solving this issue often leads to complication of the algorithm and training of additional models. In this article, we will look at an alternative approach to solving this problem.

![Developing a Replay System — Market simulation (Part 20): FOREX (I)](https://c.mql5.com/2/56/replay_p20-avatar.png)[Developing a Replay System — Market simulation (Part 20): FOREX (I)](https://www.mql5.com/en/articles/11144)

The initial goal of this article is not to cover all the possibilities of Forex trading, but rather to adapt the system so that you can perform at least one market replay. We'll leave simulation for another moment. However, if we don't have ticks and only bars, with a little effort we can simulate possible trades that could happen in the Forex market. This will be the case until we look at how to adapt the simulator. An attempt to work with Forex data inside the system without modifying it leads to a range of errors.

![Developing a Replay System — Market simulation (Part 19): Necessary adjustments](https://c.mql5.com/2/56/replay_p19_avatar.png)[Developing a Replay System — Market simulation (Part 19): Necessary adjustments](https://www.mql5.com/en/articles/11125)

Here we will prepare the ground so that if we need to add new functions to the code, this will happen smoothly and easily. The current code cannot yet cover or handle some of the things that will be necessary to make meaningful progress. We need everything to be structured in order to enable the implementation of certain things with the minimal effort. If we do everything correctly, we can get a truly universal system that can very easily adapt to any situation that needs to be handled.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zaiqwiwpkghpjnsqcfenmsojdzustrwn&ssn=1769090392686050119&ssn_dr=0&ssn_sr=0&fv_date=1769090392&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13861&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20create%20a%20simple%20Multi-Currency%20Expert%20Advisor%20using%20MQL5%20(Part%205)%3A%20Bollinger%20Bands%20On%20Keltner%20Channel%20%E2%80%94%20Indicators%20Signal%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690903925498312&fz_uniq=5048822326964887383&sv=2552)

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