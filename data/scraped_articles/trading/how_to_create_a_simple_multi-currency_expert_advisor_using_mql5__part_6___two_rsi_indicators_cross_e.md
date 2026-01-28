---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines
url: https://www.mql5.com/en/articles/14051
categories: Trading, Trading Systems
relevance_score: 15
scraped_at: 2026-01-22T17:09:15.932178
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/14051&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048825492355784552)

MetaTrader 5 / Examples


### Introduction

Multi-currency Expert Advisor is an Expert Advisor or trading robot that can trade (open orders, close orders and manage orders, for example: Trailing Stop Loss and Trailing Profit) for more than 1 symbol pair from only one symbol chart, where in this article Expert Advisor will trade for 30 pairs.

In this article we will use two RSI indicators with crossing signals, Fast RSI crossing with Slow RSI.

As has been proven in previous [articles](https://www.mql5.com/en/articles/13861), we all know that multi-currency trading is already possible with the power, capabilities and facilities provided by MQL5, both in the trading terminal and in the strategy tester.

With the aim of meeting the important needs of traders who want an efficient and effective trading robot, so by relying on the power, capabilities and facilities provided by the very reliable MQL5, we can think of various ideas and strategies to create a simple multi-currency expert advisor, where in this article we will use Two RSI indicators cross each other's lines.

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is planned to trade on a Symbol or Pair as follows:

> **For Forex:**
>
> EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP, EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD,
>
> GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,NZDJPY, CADCHF, CADJPY, CHFJPY = 28 pairs

> **Plus 2 Metal pairs:** XAUUSD (Gold) and XAGUSD (Silver)

Total is 30 pairs.

Regarding symbols or pairs that have name prefixes and/or suffixes, for the Expert Advisor in this article, I use the function to automatically handle symbol names with prefixes and/or suffixes, so that when using an EA on MetaTrader 5 from a broker with such special symbol names, everything works smoothly.

But the function to detect symbol names with prefixes and suffixes only works on Forex and Metal symbol pair names in MetaTrader 5, not on special symbols and indices.

Regarding the multi-currency expert advisor in my previous article, there are some traders who ask how to use a Multi-Currency EA as a single currency EA or as a stand-alone EA.

Actually, in the expert advisor in the previous article, there is a facility or option to use the Expert Advisor as an EA by trading only on a single currency or work as a stand alone EA.

In the Expert Advisor this time, we still use 10 choices of pairs that will be traded. One of the 10 option pairs that will be traded is "Trader's Desired Pairs", where the pairs to be traded must be entered manually by the trader in the Expert Input property. However, you must always remember that the name of the pair you enter must already be in the list of 30 pairs.

This Trader's Desired Pairs Option can actually be used so that expert advisors will only trade on single-currency or work as a stand alone EA, by only inputting one desired pair name, and then the expert advisor will only trade or work on that one pair.

The Expert Input parameter settings must be set as shown in the figure below.

![stand-alone-twp](https://c.mql5.com/2/64/stand-alone-twp.png)

In the input property example above, where the trader only inputs the name of the XAUUSD pair, then in this condition the expert advisor will only trade on the XAUUSD pair. Wherever the expert advisor is placed, among the 30 pairs available, the expert advisor will only trade on the XAUUSD pair.

Apart from that, in the expert advisor in this article, I have added an option to choose trading pair conditions, whether to trade on single-currency or trade on multi-currency.

The Expert Input Parameters need to be set as shown in the following figure.

![trading-pair-option](https://c.mql5.com/2/64/trading-pair-option.png)

![stand-alone-sp](https://c.mql5.com/2/64/stand-alone-sp.png)

In the "SP" or single-pair option, the expert advisor will only trade on the pair where the expert advisor is placed.

For example, if an expert advisor is placed on the EURUSD pair, then the expert advisor will only trade on the EURUSD pair.

So, in the expert advisors in this article, there are two ways to trade single-currency or work as a stand alone expert advisor.

> 1\. Stick to the multi-pair or "MP" option and select the Trader's Desired Pairs option, but only input one pair name, for example XAUUSD.
>
> In this option, whatever pair the expert advisor is placed on, the expert advisor will only trade on the pair name entered in Trader Wishes Pairs.

> 2\. In Select Trading Pairs Mode, select "SP" or single-pair.
>
> If an expert advisor is placed on the EURUSD pair, then the expert advisor will only trade on the EURUSD pair.

Furthermore, in the Trade on Specific Time group, options are provided for traders who want to trade in the time zone.

Maybe many traders want to trade according to the time zone, so the pairs that will trade can correspond to the time for the trading session, so in this expert advisor we still use the option for trading session (time zone).

**2\. Signal indicators.**

The relative strength index (RSI) was developed by J. Welles Wilder and published in a 1978 book, New Concepts in Technical Trading Systems, and in Commodities magazine (now Modern Trader magazine) in the June 1978 issue. It has become one of the most popular oscillator indices.

The relative strength index or RSI is a technical indicator used in the analysis of financial markets. It is intended to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period.

The Relative Strength Index Technical Indicator or [RSI](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") is a price-following oscillator that ranges between 0 and 100. When Wilder introduced the Relative Strength Index, he recommended using a 14-period RSI. Since then, the 9-period and 25-period Relative Strength Index indicators have also gained popularity.

The RSI is most commonly used on a 14-day timeframe, measured on a scale of 0 to 100, with high and low levels marked at 70 and 30, respectively. Shorter or longer timeframes are used for alternately shorter or longer outlooks. High and low levels - 80 and 20, or 90 and 10 - occur less frequently, but indicate stronger momentum.

Furthermore, in the technical analysis MetaTrader 5 it is stated that the RSI signal is used for chart analysis:

- Tops and Bottoms
- Chart Formations
- Failure Swing (Support or Resistance breakout)
- Support and Resistance levels
- Divergences

So, there are so many many variations of the RSI trading strategy.

Even an analyst and investor in a Q&A in his [article](https://www.mql5.com/go?link=https://www.quantifiedstrategies.com/rsi-trading-strategy/ "https://www.quantifiedstrategies.com/rsi-trading-strategy/"), when answering the question:

"What is the best RSI setting for day trading?

Unfortunately, RSI works best on daily bars. We have back tested a lot of data using intraday data, but it's not particularly useful."

"How do you trade with RSI?

First of all, we prefer using daily bars. Second, we prefer to use a short number of days in the settings, preferably max 5 days. Third, RSI works best on stocks and mean reverting assets with an overnight edge. We haven't been successful in forex."

However, in this article I will use signals from Fast RSI crossing Slow RSI, or RSI cross RSI for forex trading.

Whether it will be successful or not, it all still has to be experimented and tested.

An illustration of the Fast RSI signal crossing the Slow RSI line can be seen in Figure 1 and 2.

Figure 1

![RSIxRSI_signal_00](https://c.mql5.com/2/64/RSIxRSI_signal_00.png)

Figure 2

![RSIcross_variation](https://c.mql5.com/2/64/RSIcross_variation.png)

**3\. Trade & Order Management**

There are several ways to manage your trades with this multi-currency expert advisor:

3.1. Stop Loss Orders

Options: Use Order Stop Loss (Yes) or (No)

- If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.

- If the option Use Order Stop Loss (Yes): Again given the option: Use Automatic Calculation Stop Loss(Yes) or (No)

- If the option Automatic Calculation Stop Loss (Yes), then the Stop Loss calculation will be performed automatically by the Expert.

- If the option Automatic Calculation Stop Loss (No), then the trader must Input Stop Loss value in Pips.

- If the option Use Order Stop Loss (No): Then the Expert will check for each order opened, whether the signal condition is still good and order may be maintained in a profit or condition the signal has weakened and the order needs to be closed to save profit or signal condition has reversed direction and order must be closed in a loss condition.

**Note:** Especially for Close Trade and Save profit due to weak signal, an option is given, whether to activate it or not.

- If it is not activated (No), even though the signal has weakened, the order will still be maintained or will not be closed to save profit.

- If activated (Yes), the conditions for the Fast RSI and Slow RSI indicators are:

> For close the Buy orders:
>
> When the Fast RSI is above the Slow RSI, and the Fast RSI value on the current bar is smaller than the Fast RSI value on the previous bar, the Buy order will be closed.
>
> For close the Sell orders:
>
> When Fast RSI is below Slow RSI, and the Fast RSI value on the current bar is greater than the Fast RSI value on the previous bar, the Sell order will be closed.

The code to set a Stop Loss order is as follows:

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

The code to close the trade and save the profit due to a weak signal is as follows:

```
int MCEA::GetCloseInWeakSignal(const string symbol,int exis) // Signal Indicator Position Close in profit
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int bar=3;
    //--
    double rdif=rsidiff;
    double RSIFast[],
           RSISlow[];
    //--
    ArrayResize(RSIFast,bar,bar);
    ArrayResize(RSISlow,bar,bar);
    ArraySetAsSeries(RSIFast,true);
    ArraySetAsSeries(RSISlow,true);
    //--
    int x=PairsIdxArray(symbol);
    UpdatePrice(symbol,TFt);
    //--
    CopyBuffer(hRSIFast[x],0,0,bar,RSIFast);
    CopyBuffer(hRSISlow[x],0,0,bar,RSISlow);
    //--
    if(exis==down && RSIFast[1]<=RSISlow[1] && RSIFast[1]<RSIFast[2] && RSIFast[0]>RSIFast[1]+rdif) ret=rise;
    if(exis==rise && RSIFast[1]>=RSISlow[1] && RSIFast[1]>RSIFast[2] && RSIFast[0]<RSIFast[1]-rdif) ret=down;
    //--
    return(ret);
//---
  } //-end GetCloseInWeakSignal()
//---------//
```

2\. Take Profit orders.

Options: Use Order Take Profit (Yes) or (No)

- If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

- If the option Use Order Take Profit (Yes): Again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)

- If the option Automatic Calculation Order Take Profit (Yes), then the calculation of the Take Profit Order will be carried out automatically by the Expert.

- If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

The code to set a Take Profit order is as follows:

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

3\. Trailing Stop and Trailing Take Profit.

Options: Use Trailing SL/TP (Yes) or (No)

- If the Use Trailing SL/TP option is (No), then the Expert will not do trailing stop loss and trailing take profit.

- If the option Use Trailing SL/TP (Yes): Traders can choose between 2 options:

> 1\. Trailing by Price.
>
> The trailing stop will be performed by the Expert using price movements and the value in the input property, and at the same time by making trailing profit based on the variable value TPmin (minimum trailing profit value).

> 2\. Trailing By Indicator.
>
> The trailing stop will be executed by the Expert using the VIDYA indicator , and at the same time by making trailing profit based on the variable value TPmin (minimum trailing profit value).

**Note**: The Expert will carry out a trailing take profit simultaneously with a trailing stop.

According to my research and experiments, the VIDYA indicator is slightly better and ideal for trailing stops compared to the Parabolic SAR or several variants of Moving Average indicators.

Compared to the Parabolic SAR indicator, the VIDYA indicator is closer to the price movements, and compared to the AMA, DEMA and MA indicators, the VIDYA indicator is even further away from the price movements.

So in this article I decided to use the VIDYA indicator for the trailing stop function based on the indicator.

Code for the Trailing Stop Price and Indicator function:

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
            double VIDyAv[];
            ArrayResize(VIDyAv,br,br);
            ArraySetAsSeries(VIDyAv,true);
            CopyBuffer(hVIDyAv[x],0,0,br,VIDyAv);
            RefreshPrice(xsymb,TFt,br);
            //--
            if(ptype==POSITION_TYPE_BUY  && (VIDyAv[0]<mc_symbol.NormalizePrice(mc_symbol.Bid()-TSval*pip)))
               pval=VIDyAv[0];
            if(ptype==POSITION_TYPE_SELL && (VIDyAv[0]>mc_symbol.NormalizePrice(mc_symbol.Ask()+TSval*pip)))
               pval=VIDyAv[0];
            break;
          }
      }
    //--
    return(pval);
//---
  } //-end TSPrice()
//---------//
```

Modify SL/TP function code:

```
bool MCEA::ModifySLTP(const string symbx,int TS_type)
  {
//---
   ResetLastError();
   MqlTradeRequest req={};
   MqlTradeResult  res={};
   MqlTradeCheckResult check={};
   //--
   int TRSP=(Close_by_Opps==No && TS_type==1) ? 0 : TS_type;
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
               double modminsl=mc_symbol.NormalizePrice(vtrsb+((TSmin-1.0)*pip));
               double modbuysl=vtrsb;
               double modbuytp=mc_symbol.NormalizePrice(price+TPval*pip);
               bool modbuy = (price>modminsl && modbuysl>modstart && (pos_stop==0.0||modbuysl>pos_stop));
               //--
               if(modbuy && netp>minprofit)
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
               double modminsl=mc_symbol.NormalizePrice(vtrss-((TSmin+1.0)*pip));
               double modselsl=vtrss;
               double modseltp=mc_symbol.NormalizePrice(price-TPval*pip);
               bool modsel = (price<modminsl && modselsl<modstart && (pos_stop==0.0||modselsl<pos_stop));
               //--
               if(modsel && netp>minprofit)
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

In this multi-currency expert advisor, several manual button clicks are added to provide efficiency and effectiveness for traders in monitoring the expert advisor's work.

> 4.1. Set SL / TP All Orders:
>
> This button is useful if the trader has entered the parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No), but then the trader wants to use Stop Loss or Take Profit on all orders, then with a single click on the button "Set SL/TP All Orders" all orders will be modified and a Stop Loss and/or Take Profit will be applied.

> 4.2. Close All Orders:
>
> If a trader wishes to close all orders, a single click on the "Close All Orders" button will close all open orders.

> 4.3. Close All Orders Profit:

> If a trader wants to close all orders that are already profitable, a single click on the "Close All Orders Profit" button will close all open orders that are already profitable.

**5\. Management Orders and Chart Symbols.**

For multi-currency expert advisors who will trade 30 pairs from only one chart symbol, it will be very effective and efficient to provide a button panel for all symbols so that traders can change charts timeframe or symbols with just one click to see the accuracy of the indicator signal when the expert opens or closes an order.

### Implementation of planning in the MQL5 program

**1. Program header and input properties.**

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

Enumeration to select 10 option pairs to trade

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

The YN enumeration is used for the (Yes) or (No) options in the Expert Input property.

```
//--
enum YN
  {
   No,
   Yes
  };
//--
```

Enumeration to use Money Management Lot Size

```
//--
enum mmt
  {
   FixedLot,   // Fixed Lot Size
   DynamLot    // Dynamic Lot Size
  };
//--
```

Enumeration to select the timeframe to be used in calculating signal indicators

```
//--
enum TFUSE
  {
   TFM5,     // 00_PERIOD_M5
   TFM15,    // 01_PERIOD_M15
   TFM30,    // 02_PERIOD_M30
   TFH1,     // 03_PERIOD_H1
   TFH2,     // 04_PERIOD_H2
   TFH3,     // 05_PERIOD_H3
   TFH4,     // 06_PERIOD_H4
   TFH6,     // 07_PERIOD_H6
   TFH8,     // 08_PERIOD_H8
   TFH12,    // 09_PERIOD_H12
   TFD1      // 10_PERIOD_D1
  };
//--
```

With the TFUSE enumeration, I limit the use of the time frame calculations for the experts only from TF-M5 to TF-D1.

Enumeration to select the type to be used in the Trailing Stop calculation

```
//--
enum TrType
  {
    byprice, // Trailing Stop by Price
    byindi   // Trailing Stop by Indicator
  };
//--
```

Enumeration to select the type of trade the expert will trade, single currency or multi-currency.

```
//--
enum MS
 {
   SP, // Single Pair
   MP  // Multi Pairs
 };
//--
```

Expert input properties

```
//---
input group               "=== Global Strategy EA Parameter ==="; // Global Strategy EA Parameter
input TFUSE               tfinuse = TFH4;             // Select Expert TimeFrame, default PERIOD_H4
input int                 rsifast = 10;               // Input Period for Fast RSI
input ENUM_APPLIED_PRICE  frsiapp = PRICE_WEIGHTED;   // Select Fast RSI Applied Price
input int                 rsislow = 30;               // Input Period for Slow RSI
input ENUM_APPLIED_PRICE  srsiapp = PRICE_WEIGHTED;   // Select Slow RSI Applied Price
input double              rsidiff = 4.56;             // Input Differentiation between RSIs
//---
input group               "=== Select Pairs to Trade ===";  // Selected Pairs to trading
input MS                trademode = MP;              // Select Trading Pairs Mode (Multi or Single)
input PairsTrade         usepairs = All30;           // Select Pairs to Use
input string         traderwishes = "eg. eurusd,usdchf"; // If Use Trader Wishes Pairs, input pair name here, separate by comma
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
input double               SLval = 30.0;             // If Not Use Automatic SL - Input SL value in Pips
input YN                  use_tp = Yes;              // Use Order Take Profit (Yes) or (No)
input YN                  autotp = Yes;              // Use Automatic Calculation Take Profit (Yes) or (No)
input double               TPval = 60.0;             // If Not Use Automatic TP - Input TP value in Pips
input YN            TrailingSLTP = Yes;              // Use Trailing SL/TP (Yes) or (No)
input TrType               trlby = byindi;           // Select Trailing Stop Type
input double               TSval = 10.0;             // If Use Trailing Stop by Price Input value in Pips
input double               TSmin = 5.0;              // Minimum Pips to start Trailing Stop
input double               TPmin = 25.0;             // Input Trailing Profit Value in Pips
input YN           Close_by_Opps = Yes;              // Close Trade By Opposite Signal (Yes) or (No)
input YN               SaveOnRev = Yes;              // Close Trade and Save profit due to weak signal (Yes) or (No)
//--Others Expert Advisor Parameter
input group               "=== Others Expert Advisor Parameter ==="; // Others EA Parameter
input YN                  alerts = Yes;              // Display Alerts / Messages (Yes) or (No)
input YN           UseEmailAlert = No;               // Email Alert (Yes) or (No)
input YN           UseSendnotify = No;               // Send Notification (Yes) or (No)
input YN      trade_info_display = Yes;              // Select Display Trading Info on Chart (Yes) or (No)
input ulong              magicEA = 20240111;         // Expert ID (Magic Number)
//---
//---------//
```

**Note**: If the input property of Expert ID (Magic Number) is left blank, the Expert Advisor will be able to manage orders opened manually.

In the Global Strategy EA Parameters Expert Input property group, traders are instructed to select the Expert Timeframe for indicator signal calculations and enter parameters:

1. Input Period for RSI Fast and  Select RSI Fast Applied Price
2. Input Period for RSI Slow and  Select RSI Slow Applied Price
3. Input value of the differentiation between RSIs

In the Expert Advisor's "Select Pairs to Trade" property group, traders need to select the pair to trade from the 10 options provided; by default, All Forex 30 Pairs is set.

To configure the pair to be traded, we will call the HandlingSymbolArrays() function. WithHandlingSymbolArrays() function we will handle all pairs that will be traded.

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
    areur=ArraySize(EURs);
    aretc=ArraySize(JPYs);
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
        for(int t=0; t<areur; t++)
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
        case 0: // All Forex & Metal 30 Pairs
          {
            ArrayResize(DIRI,sall,sall);
            arrsymbx=sall;
            ArraySymbolResize();
            ArrayCopy(DIRI,All30,0,0,WHOLE_ARRAY);
            pairs="Multi Currency "+string(sall)+" Pairs";
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
            ArrayResize(DIRI,areur,areur);
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

Inside the HandlingSymbolArrays() function we will call the SetSymbolNamePS() function. With SetSymbolNamePS() we will be able to handle symbol names with prefixes and/or suffixes.

```
void MCEA::SetSymbolNamePS(void)
  {
//---
   int sym_Lenpre=0;
   int sym_Lensuf=0;
   string sym_pre="";
   string sym_suf="";
   SymbolSelect(Symbol(),true);
   string insymbol=Symbol();
   int inlen=StringLen(insymbol);
   int toseek=-1;
   string dep="";
   string bel="";
   string sym_use ="";
   int pairx=-1;
   string xcur[]={"EUR","GBP","AUD","NZD","USD","CAD","CHF"}; // 7 major currency
   int xcar=ArraySize(xcur);
   //--
   for(int x=0; x<xcar; x++)
     {
       toseek=StringFind(insymbol,xcur[x],0);
       if(toseek>=0)
         {
           pairx=x;
           break;
         }
     }
   if(pairx>=0)
     {
       int awl=toseek-3 <0 ? 0 : toseek-3;
       int sd=StringFind(insymbol,"SD",0);
       if(toseek==0 && sd<4)
         {
           dep=StringSubstr(insymbol,toseek,3);
           bel=StringSubstr(insymbol,toseek+3,3);
           sym_use=dep+bel;
         }
       else
       if(toseek>0)
         {
           dep=StringSubstr(insymbol,toseek,3);
           bel=StringSubstr(insymbol,toseek+3,3);
           sym_use=dep+bel;
         }
       else
         {
           dep=StringSubstr(insymbol,awl,3);
           bel=StringSubstr(insymbol,awl+3,3);
           sym_use=dep+bel;
         }
     }
   //--
   string sym_nmx=sym_use;
   int lensx=StringLen(sym_nmx);
   //--
   if(inlen>lensx && lensx==6)
     {
       sym_Lenpre=StringFind(insymbol,sym_nmx,0);
       sym_Lensuf=inlen-lensx-sym_Lenpre;
       //--
       if(sym_Lenpre>0)
         {
           sym_pre=StringSubstr(insymbol,0,sym_Lenpre);
           for(int i=0; i<xcar; i++)
             if(StringFind(sym_pre,xcur[i],0)>=0) sym_pre="";
         }
       if(sym_Lensuf>0)
         {
           sym_suf=StringSubstr(insymbol,sym_Lenpre+lensx,sym_Lensuf);
           for(int i=0; i<xcar; i++)
             if(StringFind(sym_suf,xcur[i],0)>=0) sym_suf="";
         }
     }
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

**Note**: The expert validates the pairs.  If the trader makes a mistake when entering the pair name (typos) or if the pair validation fails, the expert will receive a warning and the expert advisor will be removed from the chart.

In the expert input property group Trade on Specific Time, here the trader will choose to Trade on Specific Time Zone (Yes) or (No).

If the Trade on Specific Time Zone option is specified as No, then the expert will trade from MT5 hours 00:15 to 23:59.

If Yes, then select the enumeration options:

- Trading on Custom Session
- Trading on New Zealand Session
- Trading on Australia Sydney Session
- Trading on Asia Tokyo Session
- Trading on Europe London Session
- Trading on America New York Session

By default, Trading on Specific Time Zones is set to Yes, and is set to Trading on Custom Sessions.

Trading on Custom Session:

In this session, traders must set the time or the hours and minutes to start trading and the hours and minutes to stop trading. This means that the EA will only perform activities during the specified time period from the start to the end.

Meanwhile, in the New Zealand Trading Session to the New York US Trading Session, the time from the start of the trade to the close of the trade is calculated by the EA.

**Class for working Expert Advisor**

To build and configure the Expert Advisor workflow, we created a class that declares all the variables, objects and functions required by a multi-currency Expert Advisor.

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
    double           point;
    double           slv,
                     tpv,
                     pip,
                     xpip;
    double           SARstep,
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
    int              GetRSIxx(const string symbol);
    int              PARSAR05(const string symbol);
    int              PARSAR15(const string symbol);
    int              LotDig(const string symbol);
    //--
    bool             CheckProfit(const string symbol,ENUM_POSITION_TYPE intype);
    bool             CheckLoss(const string symbol,ENUM_POSITION_TYPE intype);
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

    //-- RSIxRSI_MCEA Config --
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
    //--
    int              hRSIFast[],
                     hRSISlow[];
    int              hVIDyAv[];
    int              hPar05[],
                     hPar15[];
    int              ALO,
                     dgts,
                     arrsar,
                     arrsymbx;
    int              sall,
                     arusd,
                     areur,
                     aretc,
                     arspc,
                     arper;
    ulong            slip;
    //--
    double           profitb[],
                     profits[];
    double           minprofit;
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
                     TFT05,
                     TFT15;
    //--
    bool             PanelExtra;
    //------------
                     MCEA(void);
                     ~MCEA(void);
    //------------
    //--
    virtual void     RSIxRSI_MCEA_Config(void);
    virtual void     ExpertActionTrade(void);
    //--
    void             ArraySymbolResize(void);
    void             CurrentSymbolSet(const string symbol);
    void             Pips(const string symbol);
    void             TradeInfo(void);
    void             Do_Alerts(const string symbx,string msgText);
    void             CheckOpenPMx(const string symbx);
    void             SetSLTPOrders(void);
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
    bool             CheckProfitLoss(const string symbol);
    bool             CloseBuyPositions(const string symbol);
    bool             CloseSellPositions(const string symbol);
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

MCEA mc;

//---------//
```

The first and most important function in the Multi-Currency Expert Advisor workflow process is called from the OnInit() is RSIxRSI\_MCEA\_Config().

The RSIxRSI\_MCEA\_Config() function configures all symbols to be used, all timeframes, all handle indicators and some important functions of the include file header for the Expert Advisor workflow.

The RSIxRSI\_MCEA\_Config() function describes and implements how to handle timeframes and create indicator handles for all indicators used in the Expert Advisor workflow.

```
//+------------------------------------------------------------------+
//| Expert Configuration                                             |
//+------------------------------------------------------------------+
void MCEA::RSIxRSI_MCEA_Config(void)
  {
//---
    //--
    HandlingSymbolArrays(); // With this function we will handle all pairs that will be traded
    //--
    TFT05=PERIOD_M5;
    TFT15=PERIOD_M15;
    ENUM_TIMEFRAMES TFs[]={PERIOD_M5,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H2,PERIOD_H3,PERIOD_H4,PERIOD_H6,PERIOD_H8,PERIOD_H12,PERIOD_D1};
    int arTFs=ArraySize(TFs);
    //--
    for(int x=0; x<arTFs; x++) if(tfinuse==x) TFt=TFs[x]; // TF for calculation signal
    //--
    //-- Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hRSISlow[x]=iRSI(DIRI[x],TFt,rsislow,srsiapp);          //-- Handle for the Slow RSI indicator
        hRSIFast[x]=iRSI(DIRI[x],TFt,rsifast,frsiapp);          //-- Handle for the Fast RSI indicator
        hVIDyAv[x]=iVIDyA(DIRI[x],TFt,9,12,0,PRICE_WEIGHTED);   //-- Handle for the VIDYA indicator for Trailing Stop
        hPar05[x]=iSAR(DIRI[x],TFT05,SARstep,SARmaxi);          //-- Handle for the iSAR indicator for M5 Timeframe
        hPar15[x]=iSAR(DIRI[x],TFT15,SARstep,SARmaxi);          //-- Handle for the iSAR indicator for M15 Timeframe
        //--
      }
    //--
    minprofit=NormalizeDouble(TSmin/100.0,2);
    //--
    ALO=(int)mc_account.LimitOrders()>sall ? sall : (int)mc_account.LimitOrders();
    if(Close_by_Opps==No)
      {
        if((int)mc_account.LimitOrders()>=(sall*2)) ALO=sall*2;
        else
        ALO=(int)(mc_account.LimitOrders()/2);
      }
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
  } //-end RSIxRSI_MCEA_Config()
//---------//
```

**2\. Expert tick function**

Within the Expert Tick function (OnTick() function) we will call one of the most important functions in a multi-currency Expert Advisor, namely ExpertActionTrade() function. The whole process of EA working for trading is included in this function.

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

It means that the ExpertActionTrade() function will perform all activities and manage automatic trading, starting from opening orders, closing orders, trailing stops or trailing profits and other additional activities.

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
            switch(trademode)
              {
                case SP:
                  {
                    if(mc.DIRI[x]!=Symbol()) continue;
                    symbol=Symbol();
                    mc.pairs=mc.pairs+" ("+symbol+")";
                    break;
                  }
                case MP:
                  {
                    if(mc.DIRI[x]==Symbol()) symbol=Symbol();
                    else symbol=mc.DIRI[x];
                    break;
                  }
              }
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
                    if(mc.profitb[x]>mc.minprofit && mc.xob[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Buy)==mc.Sell)
                      {
                        mc.CloseBuyPositions(symbol);
                        mc.Do_Alerts(symbol,"Close BUY order "+symbol+" to save profit due to weak signal.");
                      }
                    if(mc.profits[x]>mc.minprofit && mc.xos[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Sell)==mc.Buy)
                      {
                        mc.CloseSellPositions(symbol);
                        mc.Do_Alerts(symbol,"Close SELL order "+symbol+" to save profit due to weak signal.");
                      }
                  }
                //--
                if(TrailingSLTP==Yes) //-- Use Trailing SL/TP (Yes)
                  {
                    mc.ModifySLTP(symbol,trlby);
                  }
              }
            //--
            mc.CheckOpenPMx(symbol);
            if(Close_by_Opps==No && (mc.xob[x]+mc.xos[x]>1))
              {
                mc.CheckProfitLoss(symbol);
                mc.Do_Alerts(symbol,"Close order due stop in loss.");
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

Meanwhile, the Day Trading On/Off property group allows traders to trade on certain days from Sunday to Saturday. This allows traders to enable or disable experts to trade on a particular day using the (Yes) or (No) options.

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

The execution for the Day Trading On/Off is as follows:

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

**Notes**: Day Trading On/Off conditions will be displayed in the Trading Info on Chart.

Which is executed by the TradingDay() function

```
string MCEA::TradingDay(void)
  {
//---
   int trdday=ThisTime(dow);
   switch(trdday)
     {
        case 0: daytrade="Sunday";    break;
        case 1: daytrade="Monday";    break;
        case 2: daytrade="Tuesday";   break;
        case 3: daytrade="Wednesday"; break;
        case 4: daytrade="Thursday";  break;
        case 5: daytrade="Friday";    break;
        case 6: daytrade="Saturday";  break;
     }
   return(daytrade);
//---
  } //-end TradingDay()
//---------//
```

Traders have the option to trade by time zone in the Expert Advisor's "Trade at Specific Time" property group.

**Notes**: As explained above, in the case of trading on the New Zealand Session to trading on the US New York Session, the time from the start of trading to the end of trading is calculated by the EA.

Therefore, in the Expert Entry properties, traders only need to set the time for the hour and minute when trading starts and the time for the hour and minute when trading ends for the Custom Session.

The ExpertActionTrade() function has been extended with a call to the boolean Trade\_session() function specifically for time zone trading.

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

The workflow in Trade at Specific Time is as follows, first the ExpertActionTrade() function calls the Trade\_session() function, then the Trade\_session() function calls the Set\_Time\_Zone() function, and finally the Set\_Time\_Zone() function calls the Time\_Zone() function.

```
void MCEA::Set_Time_Zone(void)
  {
//---
    //-- Server Time==TimeCurrent()
    datetime TTS=TimeTradeServer();
    datetime GMT=TimeGMT();
    //--
    MqlDateTime svrtm,gmttm;
    TimeToStruct(TTS,svrtm);
    TimeToStruct(GMT,gmttm);
    int svrhr=svrtm.hour;  // Server time hour
    int gmthr=gmttm.hour;  // GMT time hour
    int difhr=svrhr-gmthr; // Time difference Server time to GMT time
    //--
    int NZSGMT=12;  // New Zealand Session GMT/UTC+12
    int AUSGMT=10;  // Australia Sydney Session GMT/UTC+10
    int TOKGMT=9;   // Asia Tokyo Session GMT/UTC+9
    int EURGMT=0;   // Europe London Session GMT/UTC 0
    int USNGMT=-5;  // US New York Session GMT/UTC-5
    //--
    int NZSStm=8;   // New Zealand Session time start: 08:00 Local Time
    int NZSCtm=17;  // New Zealand Session time close: 17:00 Local Time
    int AUSStm=7;   // Australia Sydney Session time start: 07:00 Local Time
    int AUSCtm=17;  // Australia Sydney Session time close: 17:00 Local Time
    int TOKStm=9;   // Asia Tokyo Session time start: 09:00 Local Time
    int TOKCtm=18;  // Asia Tokyo Session time close: 18:00 Local Time
    int EURStm=9;   // Europe London Session time start: 09:00 Local Time
    int EURCtm=19;  // Europe London Session time close: 19:00 Local Time
    int USNStm=8;   // US New York Session time start: 08:00 Local Time
    int USNCtm=17;  // US New York Session time close: 17:00 Local Time
    //--
    int nzo = (NZSStm+difhr-NZSGMT)<0 ? 24+(NZSStm+difhr-NZSGMT) : (NZSStm+difhr-NZSGMT);
    int nzc = (NZSCtm+difhr-NZSGMT)<0 ? 24+(NZSCtm+difhr-NZSGMT) : (NZSCtm+difhr-NZSGMT);
    //--
    int auo = (AUSStm+difhr-AUSGMT)<0 ? 24+(AUSStm+difhr-AUSGMT) : (AUSStm+difhr-AUSGMT);
    int auc = (AUSCtm+difhr-AUSGMT)<0 ? 24+(AUSCtm+difhr-AUSGMT) : (AUSCtm+difhr-AUSGMT);
    //--
    int tko = (TOKStm+difhr-TOKGMT)<0 ? 24+(TOKStm+difhr-TOKGMT) : (TOKStm+difhr-TOKGMT);
    int tkc = (TOKCtm+difhr-TOKGMT)<0 ? 24+(TOKCtm+difhr-TOKGMT) : (TOKCtm+difhr-TOKGMT);
    //--
    int euo = (EURStm+difhr-EURGMT)<0 ? 24+(EURStm+difhr-EURGMT) : (EURStm+difhr-EURGMT);
    int euc = (EURCtm+difhr-EURGMT)<0 ? 24+(EURCtm+difhr-EURGMT) : (EURCtm+difhr-EURGMT);
    //--
    int uso = (USNStm+difhr-USNGMT)<0 ? 24+(USNStm+difhr-USNGMT) : (USNStm+difhr-USNGMT);
    int usc = (USNCtm+difhr-USNGMT)<0 ? 24+(USNCtm+difhr-USNGMT) : (USNCtm+difhr-USNGMT);
    if(usc==0||usc==24) usc=23;
    //--
    //---Trading on Custom Session
    int _days00=ThisTime(day);
    int _days10=ThisTime(day);
    if(stsescuh>clsescuh) _days10=ThisTime(day)+1;
    tmopcu=ReqDate(_days00,stsescuh,stsescum);
    tmclcu=ReqDate(_days10,clsescuh,clsescum);
    //--
    //--Trading on New Zealand Session GMT/UTC+12
    int _days01=ThisTime(hour)<nzc ? ThisTime(day)-1 : ThisTime(day);
    int _days11=ThisTime(hour)<nzc ? ThisTime(day) : ThisTime(day)+1;
    tmop01=ReqDate(_days01,nzo,0);    // start: 08:00 Local Time == 20:00 GMT/UTC
    tmcl01=ReqDate(_days11,nzc-1,59); // close: 17:00 Local Time == 05:00 GMT/UTC
    //--
    //--Trading on Australia Sydney Session GMT/UTC+10
    int _days02=ThisTime(hour)<auc ? ThisTime(day)-1 : ThisTime(day);
    int _days12=ThisTime(hour)<auc ? ThisTime(day) : ThisTime(day)+1;
    tmop02=ReqDate(_days02,auo,0);    // start: 07:00 Local Time == 21:00 GMT/UTC
    tmcl02=ReqDate(_days12,auc-1,59); // close: 17:00 Local Time == 07:00 GMT/UTC
    //--
    //--Trading on Asia Tokyo Session GMT/UTC+9
    int _days03=ThisTime(hour)<tkc ? ThisTime(day) : ThisTime(day)+1;
    int _days13=ThisTime(hour)<tkc ? ThisTime(day) : ThisTime(day)+1;
    tmop03=ReqDate(_days03,tko,0);    // start: 09:00 Local Time == 00:00 GMT/UTC
    tmcl03=ReqDate(_days13,tkc-1,59); // close: 18:00 Local Time == 09:00 GMT/UTC
    //--
    //--Trading on Europe London Session GMT/UTC 00:00
    int _days04=ThisTime(hour)<euc ? ThisTime(day) : ThisTime(day)+1;
    int _days14=ThisTime(hour)<euc ? ThisTime(day) : ThisTime(day)+1;
    tmop04=ReqDate(_days04,euo,0);     // start: 09:00 Local Time == 09:00 GMT/UTC
    tmcl04=ReqDate(_days14,euc-1,59);  // close: 19:00 Local Time == 19:00 GMT/UTC
    //--
    //--Trading on US New York Session GMT/UTC-5
    int _days05=ThisTime(hour)<usc  ? ThisTime(day) : ThisTime(day)+1;
    int _days15=ThisTime(hour)<=usc ? ThisTime(day) : ThisTime(day)+1;
    tmop05=ReqDate(_days05,uso,0);  // start: 08:00 Local Time == 13:00 GMT/UTC
    tmcl05=ReqDate(_days15,usc,59); // close: 17:00 Local Time == 22:00 GMT/UTC
    //--
    //--Not Use Trading Time Zone
    if(trd_time_zone==No)
      {
        tmopno=ReqDate(ThisTime(day),0,15);
        tmclno=ReqDate(ThisTime(day),23,59);
      }
    //--
    Time_Zone();
    //--
    return;
//---
  } //-end Set_Time_Zone()
//---------//
```

```
void MCEA::Time_Zone(void)
  {
//---
   //--
   tz_ses="";
   //--
   switch(session)
     {
       case Cus_Session:
         {
           SesCuOp=StringToTime(tmopcu);
           SesCuCl=StringToTime(tmclcu);
           zntm=SesCuOp;
           znop=SesCuOp;
           zncl=SesCuCl;
           tz_ses="Custom_Session";
           tz_opn=timehr(stsescuh,stsescum);
           tz_cls=timehr(clsescuh,clsescum);
           break;
         }
       case New_Zealand:
         {
           Ses01Op=StringToTime(tmop01);
           Ses01Cl=StringToTime(tmcl01);
           zntm=Ses01Op;
           znop=Ses01Op;
           zncl=Ses01Cl;
           tz_ses="New_Zealand/Oceania";
           tz_opn=timehr(ReqTime(Ses01Op,hour),ReqTime(Ses01Op,min));
           tz_cls=timehr(ReqTime(Ses01Cl,hour),ReqTime(Ses01Cl,min));
           break;
         }
       case Australia:
         {
           Ses02Op=StringToTime(tmop02);
           Ses02Cl=StringToTime(tmcl02);
           zntm=Ses02Op;
           znop=Ses02Op;
           zncl=Ses02Cl;
           tz_ses="Australia Sydney";
           tz_opn=timehr(ReqTime(Ses02Op,hour),ReqTime(Ses02Op,min));
           tz_cls=timehr(ReqTime(Ses02Cl,hour),ReqTime(Ses02Cl,min));
           break;
         }
       case Asia_Tokyo:
         {
           Ses03Op=StringToTime(tmop03);
           Ses03Cl=StringToTime(tmcl03);
           zntm=Ses03Op;
           znop=Ses03Op;
           zncl=Ses03Cl;
           tz_ses="Asia/Tokyo";
           tz_opn=timehr(ReqTime(Ses03Op,hour),ReqTime(Ses03Op,min));
           tz_cls=timehr(ReqTime(Ses03Cl,hour),ReqTime(Ses03Cl,min));
           break;
         }
       case Europe_London:
         {
           Ses04Op=StringToTime(tmop04);
           Ses04Cl=StringToTime(tmcl04);
           zntm=Ses04Op;
           znop=Ses04Op;
           zncl=Ses04Cl;
           tz_ses="Europe/London";
           tz_opn=timehr(ReqTime(Ses04Op,hour),ReqTime(Ses04Op,min));
           tz_cls=timehr(ReqTime(Ses04Cl,hour),ReqTime(Ses04Cl,min));
           break;
         }
       case US_New_York:
         {
           Ses05Op=StringToTime(tmop05);
           Ses05Cl=StringToTime(tmcl05);
           zntm=Ses05Op;
           znop=Ses05Op;
           zncl=Ses05Cl;
           tz_ses="US/New_York";
           tz_opn=timehr(ReqTime(Ses05Op,hour),ReqTime(Ses05Op,min));
           tz_cls=timehr(ReqTime(Ses05Cl,hour),ReqTime(Ses05Cl,min));
           break;
         }
     }
   //--
   if(trd_time_zone==No)
     {
       SesNoOp=StringToTime(tmopno);
       SesNoCl=StringToTime(tmclno);
       zntm=SesNoOp;
       znop=SesNoOp;
       zncl=SesNoCl;
       tz_ses="Not Use Time Zone";
       tz_opn=timehr(ReqTime(SesNoOp,hour),ReqTime(SesNoOp,min));
       tz_cls=timehr(ReqTime(SesNoCl,hour),ReqTime(SesNoCl,min));
     }
   //--
   return;
//---
  } //-end Time_Zone()
//---------//
```

**3\. How to get trading signals for open positions?**

In order to get a signal to open a position, the ExpertActionTrade() function calls the GetOpenPosition() function.

```
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int rsix=GetRSIxx(symbol);
    int par15=PARSAR15(symbol);
    //--
    if(rsix==rise && par15==rise) ret=rise;
    if(rsix==down && par15==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetOpenPosition()
//---------//
```

And the GetOpenPosition() function will call 2 functions:

1. GetRSIxx() function which takes the buffer value of 2 RSI and calculates the signal algorithm.
2. PARSAR15() function as a filter.

```
int MCEA::GetRSIxx(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int bar=3;
    //--
    double rdif=rsidiff;
    double RSIFast[],
           RSISlow[];
    //--
    ArrayResize(RSIFast,bar,bar);
    ArrayResize(RSISlow,bar,bar);
    ArraySetAsSeries(RSIFast,true);
    ArraySetAsSeries(RSISlow,true);
    //--
    int x=PairsIdxArray(symbol);
    UpdatePrice(symbol,TFt);
    //--
    CopyBuffer(hRSIFast[x],0,0,bar,RSIFast);
    CopyBuffer(hRSISlow[x],0,0,bar,RSISlow);
    //--
    if(RSIFast[1]<=RSISlow[1] && RSIFast[0]>RSISlow[0]+rdif) ret=rise;
    if(RSIFast[1]>=RSISlow[1] && RSIFast[0]<RSISlow[0]-rdif) ret=down;
    //--
    return(ret);
//---
  } //-end GetRSIxx()
//---------//
```

```
int MCEA::PARSAR15(const string symbol) // formula Parabolic SAR M5
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

Inside the GetRSIxx() function and PARSAR15() function, we use and call 1 function, which is thePairsIdxArray() function.

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

The PairsIdxArray() function is used to get the name of the requested symbol and the handles of its indicators. Then the corresponding indicator handle is called to get the buffer value of the RSI indicator and PSAR indicator from that symbol and timeframe.

In this Expert Advisor we will use 2 RSI indicators.

They have different input parameters:

Fast RSI:

- symbol = according to the requested symbol,
- timeframe = as specified in Expert timeframe.
- ma\_period = 10, according to Input Period for Fast RSI
- applied\_price = PRICE\_WEIGHTED, according to Fast RSI Applied Price

Slow RSI:

- symbol = according to the requested symbol,
- timeframe = as specified in Expert timeframe.
- ma\_period = 30, according to Input Period for Slow RSI
- applied\_price = PRICE\_WEIGHTED, according to Slow RSI Applied Price

```
    //-- Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hRSISlow[x]=iRSI(DIRI[x],TFt,rsislow,srsiapp);          //-- Handle for the Slow RSI indicator
        hRSIFast[x]=iRSI(DIRI[x],TFt,rsifast,frsiapp);          //-- Handle for the Fast RSI indicator
        hVIDyAv[x]=iVIDyA(DIRI[x],TFt,9,12,0,PRICE_WEIGHTED);   //-- Handle for the VIDYA indicator for Trailing Stop
        hPar05[x]=iSAR(DIRI[x],TFT05,SARstep,SARmaxi);          //-- Handle for the iSAR indicator for M5 Timeframe
        //--
      }
```

So, to get the buffer value for each RSI indicator, we will copy each buffer from the indicators handle.

To copy the Fast RSI buffer (buffer 0) from the Fast RSI handle to the destination array:

```
CopyBuffer(hRSIFast[x],0,0,bar,RSIFast);
```

To copy the Slow RSI buffer (buffer 0) from the Slow RSI handle to the destination array:

```
CopyBuffer(hRSISlow[x],0,0,bar,RSISlow);
```

After executing the 2 functions GetRSIxx() and PARSAR05(), the GetOpenPosition() function will provide the values:

- Value 0, signal unknown.
- Value 1, is a signal for open Buy order;
- Value -1, is a signal for open Sell order.

When the GetOpenPosition() function returns a value of 1, the Expert Advisor calls the OpenBuy() function to open a Buy order.

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

Meanwhile, if the GetOpenPosition() function returns a value of -1, the Expert Advisor will call the OpenSell() function to open s Sell order.

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

**4\. ChartEvent Function**

To support effectiveness and efficiency in using Multi-Currency Expert Advisors, it is considered necessary to create one or more manual buttons in managing orders and changing charts timeframe or symbols.

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

In the Other Expert Advisor Parameters group of input properties, the trader is given the opportunity to select whether to display trading information on the chart (Yes) or (No).

If this option is selected (Yes), trade information will be displayed on the chart to which the Expert Advisor is attached by calling the TradeInfo() function.

We have also added a function to describe the time according to trading time zone conditions as part of the TradeInfo() function.

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

The Multi-Currency Expert Advisor RSIxRSI\_MCEA interface resembles the following figure.

![RSIxRSI_Look](https://c.mql5.com/2/64/RSIxRSI_Look.png)

As you can see, there are buttons "M", "C" and "R" under the Expert Advisor name RSIxRSI\_MCEA.

When the M button is clicked, a panel of manual clicking buttons is displayed as shown below.

![Expert_manual_button_01](https://c.mql5.com/2/64/Expert_manual_button_01.png)

The trader can manage orders manually when the manual click button panel is displayed:

1\. Set SL/TP All Orders

Set SL/TP All Orders As explained above, if the trader enters the parameters Use Order Stop Loss (No) and/or Use Order Take Profit (No), but then the trader wants to use Stop Loss or Take Profit on all orders, then a single click on the "Set SL / TP All Orders" button will modify all orders and apply Stop Loss and/or Take Profit.

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

2\. Close All Orders

If a trader wishes to close all orders, a single click on the "Close All Orders" button will close all open orders.

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

3\. Close All Profits

If a trader wishes to close all orders that are already profitable, a single click on the "Close All Profits" button will close all open orders that are already profitable.

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

When the C button is clicked, a panel button with 30 symbol names or pairs is displayed and traders can click on one of the pair names or symbol names.

Clicking on one of the pair names or symbol names immediately replaces the chart symbol with the symbol whose name was clicked.

![Expert_manual_button_02](https://c.mql5.com/2/64/Expert_manual_button_02.png)

```
void CreateSymbolPanel()
  {
//---
    //--
    ResetLastError();
    DeletePanelButton();
    int sydis=83;
    int tsatu=int(mc.sall/2);
    //--
    CreateButtonTemplate(0,"Template",180,367,STYLE_SOLID,5,BORDER_RAISED,clrYellow,clrBurlyWood,clrWhite,CORNER_RIGHT_UPPER,187,45,true);
    CreateButtonTemplate(0,"TempCCS",167,25,STYLE_SOLID,5,BORDER_RAISED,clrYellow,clrBlue,clrWhite,CORNER_RIGHT_UPPER,181,50,true);
    CreateButtonClick(0,"X",14,14,"Arial Black",10,BORDER_FLAT,"X",clrWhite,clrWhite,clrRed,ANCHOR_CENTER,CORNER_RIGHT_UPPER,22,48,true,"Close Symbol Panel");
    //--
    string chsym="Change SYMBOL";
    int cspos=int(181/2)+int(StringLen(chsym)/2);
    CreateButtontLable(0,"CCS","Bodoni MT Black",chsym,11,clrWhite,ANCHOR_CENTER,CORNER_RIGHT_UPPER,cspos,62,true,"Change Chart Symbol");
    //--
    for(int i=0; i<tsatu; i++)
      CreateButtonClick(0,mc.AS30[i],80,17,"Bodoni MT Black",8,BORDER_RAISED,mc.AS30[i],clrYellow,clrBlue,clrWhite,ANCHOR_CENTER,CORNER_RIGHT_UPPER,180,sydis+(i*22),true,"Change to "+mc.AS30[i]);
    //--
    for(int i=tsatu; i<mc.sall; i++)
      CreateButtonClick(0,mc.AS30[i],80,17,"Bodoni MT Black",8,BORDER_RAISED,mc.AS30[i],clrYellow,clrBlue,clrWhite,ANCHOR_CENTER,CORNER_RIGHT_UPPER,94,sydis+((i-tsatu)*22),true,"Change to "+mc.AS30[i]);
    //--
    ChartRedraw(0);
    //--
    return;
//---
   } //-end CreateSymbolPanel()
//---------//
```

In this case, the OnChartEvent() function will be called the ChangeChartSymbol() function when one of the symbol names is clicked.

```
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
       int lensymbol=StringLen(Symbol());
       int lensparam=StringLen(sparam);

       //--- if Symbol button is click
       if(lensparam==lensymbol)
         {
           int sx=mc.ValidatePairs(sparam);
           ChangeChartSymbol(mc.AS30[sx],CCS);
           mc.PanelExtra=false;
         }
       //--
     }
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

And finally, clicking on the R button will remove the Multi-Currency Expert Advisor RSIxRSI\_MCEA from the chart, so traders don't have to remove the experts manually.

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

The advantage of MetaTrader 5 Strategy Tester is that it supports and allows us to test strategies that will trade on multiple symbols or test automatic trading for all available symbols and on all available timeframes.

Therefore, on the MetaTrader 5 Strategy Tester platform we will test the RSIxRSI\_MCEA Multi-Currency Expert Advisor.

In the first test, we have placed the RSIxRSI\_MCEA on the XAGUSD pair and the H4 timeframe, with acustom time period of 2023.10.01 to 2024.01.05

The test was carried out with two different input properties, specifically in the Global Strategy EA Parameter group and Trade & Order Management parameters group.

1\. The RSIxRSI\_MCEA on the XAGUSD pair and the H4 timeframe

![ST_07_e1](https://c.mql5.com/2/64/ST_07_e1.png)

![ST_07_e2](https://c.mql5.com/2/64/ST_07_e2.png)

The results of the first test are as in the image below.

![ST_07_e3](https://c.mql5.com/2/64/ST_07_e3.png)

2\. The RSIxRSI\_MCEA on the XAGUSD pair and the H12 timeframe

![ST_08_e1](https://c.mql5.com/2/64/ST_08_e1.png)

![ST_08_e2](https://c.mql5.com/2/64/ST_08_e2.png)

The results of the second test are as in the image below.

![ST_08_e3](https://c.mql5.com/2/64/ST_08_e3.png)

### Conclusion

The conclusion in creating a Multi-Currency Expert Advisor with signals from Fast RSI crossing Slow RSI, or RSI cross RSI for forex trading using MQL5 is as follows:

1. It turns out that creating a multi-currency Expert Advisor in MQL5 is very simple and not much differentfrom creating a single-currency Expert Advisor.
2. Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders bye liminating the need to open many chart symbols for trading.
3. Applying the right trading strategy will increase the probability of profit compared to using a singlecurrency Expert Advisor. This is because losses in one pair will be covered by profits in other pairs.
4. This RSIxRSI\_MCEA Multi-Currency Expert Advisor is just a sample for learning and idea generation. The test results on the Strategy Tester are still not good. Therefore, by experimenting and testing on different timeframes or different indicator period calculations, it is possible to get better strategy and more profitable results.
5. In my opinion, this RSI cross RSI strategy should be further researched with various different experiments, starting from timeframe, fast period RSI, slow period RSI, differentiation value of both RSIs.
Based on the results of my experiments on Strategy Tester, on timeframes below H4 the results are not good, only on timeframes H4 and above, for example on timeframe H8 and H12, the results are good with few open trades, compared to small timeframes with many open trades, but in loss.

We hope that this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders to learn and develop ideas.

Thanks for reading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14051.zip "Download all attachments in the single ZIP archive")

[RSIxRSI\_MCEA.mq5](https://www.mql5.com/en/articles/download/14051/rsixrsi_mcea.mq5 "Download RSIxRSI_MCEA.mq5")(112.24 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal](https://www.mql5.com/en/articles/14329)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR](https://www.mql5.com/en/articles/13008)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/461178)**
(11)


![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
28 Mar 2024 at 16:04

**liane.blane [#](https://www.mql5.com/en/forum/461178#comment_52865195):**

Thank you Roberto for this EA of the two crossing RSI's. I tried attaching to my MT4 programme however I was not successful. Is this EA built for MT5?

From the title you can read that this program was created using MQL5.

![Achmad Muchtar](https://c.mql5.com/avatar/2023/7/64A81303-C860.jpg)

**[Achmad Muchtar](https://www.mql5.com/en/users/amsb17)**
\|
16 Apr 2025 at 03:23

Wow, good job. I myself stick to MISS and KISS in coding. And, the result in

Live trading is in line with expectation.

Keep up the good work bro.👍

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
16 Apr 2025 at 04:23

Very useful article with code sections, multisymbol and buttons - I will study more and apply in my robots.

The [trading approach](https://www.mql5.com/en/market "A Market of Applications for the MetaTrader 5 and MetaTrader 4") is also ok - I need to unpimise and trade!

Thank you, Roberto, for a competent article.

It is very useful for me as a trader and programmer!

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
27 Apr 2025 at 16:27

**Roman Shiredchenko [#](https://www.mql5.com/en/forum/461178#comment_56467825):**

Very useful article with code sections, multisymbol and buttons - I will study more and apply in my robots.

The [trading approach](https://www.mql5.com/en/market "A Market of Applications for the MetaTrader 5 and MetaTrader 4") is also ok - I need to unpimise and trade!

Thank you, Roberto, for a competent article.

It is very useful for me as a trader and programmer!

You are welcome.

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
27 Apr 2025 at 16:29

**Roberto Jacobs [#](https://www.mql5.com/ru/forum/467313#comment_56560336):**

Thank you, Roberto, for a knowledgeable article.

It is very useful for me as a trader and programmer!

You are welcome.

You're welcome

![ALGLIB numerical analysis library in MQL5](https://c.mql5.com/2/58/ALGLIB_in_MQL5_avatar.png)[ALGLIB numerical analysis library in MQL5](https://www.mql5.com/en/articles/13289)

The article takes a quick look at the ALGLIB 3.19 numerical analysis library, its applications and new algorithms that can improve the efficiency of financial data analysis.

![Building and testing Aroon Trading Systems](https://c.mql5.com/2/64/Building_and_testing_Aroon_Trading_Systems___LOGO.png)[Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

In this article, we will learn how we can build an Aroon trading system after learning the basics of the indicators and the needed steps to build a trading system based on the Aroon indicator. After building this trading system, we will test it to see if it can be profitable or needs more optimization.

![Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://c.mql5.com/2/65/Introduction_to_MQL5_rPart_38_Mastering_the_Core_Elements_of_MQL5____LOGO___small-transformed.png)[Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://www.mql5.com/en/articles/14099)

Explore the fundamentals of MQL5 programming in this beginner-friendly article, where we demystify arrays, custom functions, preprocessors, and event handling, all explained with clarity making every line of code accessible. Join us in unlocking the power of MQL5 with a unique approach that ensures understanding at every step. This article sets the foundation for mastering MQL5, emphasizing the explanation of each line of code, and providing a distinct and enriching learning experience.

![Deep Learning Forecast and ordering with Python and MetaTrader5 python package and ONNX model file](https://c.mql5.com/2/64/Deep_Learning_Forecast_and_ordering_with_Python_and_MetaTrader5_python_packag___LOGOe.png)[Deep Learning Forecast and ordering with Python and MetaTrader5 python package and ONNX model file](https://www.mql5.com/en/articles/13975)

The project involves using Python for deep learning-based forecasting in financial markets. We will explore the intricacies of testing the model's performance using key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) and we will learn how to wrap everything into an executable. We will also make a ONNX model file with its EA.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lddeoswjawkivjzwqyjechyifekwkmwg&ssn=1769090409322393163&ssn_dr=0&ssn_sr=0&fv_date=1769090409&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14051&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20create%20a%20simple%20Multi-Currency%20Expert%20Advisor%20using%20MQL5%20(Part%206)%3A%20Two%20RSI%20indicators%20cross%20each%20other%27s%20lines%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909040925821606&fz_uniq=5048825492355784552&sv=2552)

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