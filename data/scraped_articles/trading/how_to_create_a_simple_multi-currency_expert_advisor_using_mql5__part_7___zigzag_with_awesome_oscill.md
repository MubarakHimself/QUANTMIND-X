---
title: How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 7): ZigZag with Awesome Oscillator Indicators Signal
url: https://www.mql5.com/en/articles/14329
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 16
scraped_at: 2026-01-22T17:08:38.673271
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14329&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5048821115784109904)

MetaTrader 5 / Examples


### Introduction

The Multi-currency Expert Advisor is a automated trading that can open, close, and manage orders for more than one symbol pair from a single symbol chart.

This article focuses on the Expert Advisor trading for 30 pairs and using the ZigZag indicator, which is filtered with the Awesome Oscillator or filters each other's signals.

As demonstrated in previous articles, multi-currency trading is possible using MQL5's power, capabilities, and facilities in both the trading terminal and strategy tester.

In order to meet the needs of traders seeking an efficient and effective automated trading, we rely on the power, capabilities, and facilities provided by the reliable MQL5. Our goal is to create a simple multi-currency expert advisor using various ideas and strategies. This article will focus on using the ZigZag indicator, which will be filtered with the Awesome Oscillator or by filtering each other's signals.

### Plans and Features

**1\. Trading Currency Pairs.**

This Multi-Currency Expert Advisor is planned to trade on a Symbol or Pair as follows:

> **For Forex:**
>
> EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCAD,USDCHF,USDJPY,EURGBP,EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD,
>
> GBPCAD,GBPCHF,GBPJPY,AUDNZD,AUDCAD,AUDCHF,AUDJPY,NZDCAD,NZDCHF,NZDJPY, CADCHF, CADJPY, CHFJPY = **28 pairs**

> **Plus 2 Metal pairs:** XAUUSD (Gold) and XAGUSD (Silver)
>
> **Total is 30 pairs.**

To ensure smooth operation of the Expert Advisor discussed in this article, I have implemented a function that automatically handles symbol names with prefixes and/or suffixes.

However, it is important to note that this function only works for Forex and Metal symbol pair names in MT5, and not for special symbols and indices.

1.1. Additional features:

1.1.1. Single Currency Expert Advisor

Some users have inquired about using this Multi-Currency EA as a single currency or stand-alone EA.

To address this, a feature has been added to allow for the use of this Multi-Currency EA as a Single-Currency EA.

In the Select Trading Pairs Mode (Multi or Single) options, traders can choose between two trading pair conditions: single-currency or multi-currency.

> 1\. Single Pair (single-currency)
>
> 2\. Multi Pairs (multi-currency)

Set the Expert Input Parameters as shown in the figure below

![trading-pair-option](https://c.mql5.com/2/70/trading-pair-option.png)

If 'Single Pair' is selected:

The 'SP' or single-pair option limits the expert advisor to trading only on the pair where it is placed.

For instance, if an expert advisor is placed on the EURUSD pair, it will only trade on the EURUSD pair.

If 'Multi Pairs' is selected:

The option group 'Selected Pairs to Trade' includes 10 pairs of options that will be traded.

One of these pairs is called 'Trader's Wishes Pairs', which requires the trader to manually enter the pairs to be traded in the 'Expert Input' property.

However, it is important to remember that the name of the pair entered must already be on the list of 30 pairs.

The Trader's Wishes Pairs Option can be used to ensure that expert advisors only trade on single currencies or work as a standalone EA.

The expert advisor will only trade or work on the desired pair name provided. This ensures a focused approach to trading or working on a single pair.

If the trader only inputs the name of the XAUUSD pair, the expert advisor will only trade on that pair.

The expert advisor will only trade on the XAUUSD pair, regardless of its placement among the 30 available pairs.

The settings for the Expert Input parameter must be configured as depicted in the figure below.

![stand-alone-twp](https://c.mql5.com/2/70/stand-alone-twp.png)

So, in this article, there are two ways to trade single-currency or work as a stand-alone expert advisor using the expert advisors.

> 1\. Stick to the multi-pair or 'MP' option and select the 'Trader's Desired Pairs' option. However, only input one pair name, such as XAUUSD.
>
> This option restricts the expert advisor to trading only on the pair specified in Trader Wishes Pairs. It will not trade on any other pairs.
>
> 2\. In Select Trading Pairs Mode, choose 'SP' or single-pair.
>
> If an expert advisor is applied to the EURUSD pair, it will exclusively trade on the EURUSD pair.

1.1.2. Trade on Specific Time

In the Trade on Specific Time group, options are provided for traders who want to trade in the time zone.

Maybe many traders want to trade according to the time zone, so the pairs that will trade can correspond to the time for the trading session, so in this expert advisor we still use the option for trading session (time zone).

**2\. Signal indicators.**

2.1. ZigZag Indicator.

The ZigZag indicator is a method of measuring price movements without unnecessary noise. It operates by determining the distance between price swings (highs and lows). Subsequently, this indicator computes the pullback. If the pullback exceeds a certain anticipated amount, the price movement is deemed to be complete.

As is known, ZigZag is one of the oldest technical indicators that made its way into currency trading from the stock market. It enables traders to visualize the market's structure.

When trying to assess price changes, the ZigZag indicator assists traders in automatically filtering out minor price movements.

Utilizing the ZigZag indicator can aid in obtaining a clearer understanding of the market's structure.

When price movement conditions fluctuate with the impact of random price fluctuations, the ZigZag indicator can be used to help identify price trends and changes in price trends.

Technical analysts say that:

- The Zig Zag indicator reduces the impact of random price fluctuations and is used to identify price trends and changes in price trends.
- The indicator reduces noise levels, emphasizing underlying trends higher and lower.
- The Zig Zag indicator works best in strongly trending markets.

> Zig Zag Indicator Limitations.
>
> Similar to other trend-following indicators, buy and sell signals are determined by looking at past price movements which may not accurately predict future price movements. For instance, most of a trend could have already occurred by the time a Zig Zag line is generated.

> Traders should be aware that the most recent Zig Zag line may not be permanent. When price changes direction, the indicator starts to draw a new line.
>
> If that line does not reach the indicator’s percentage setting and the price of the security reverses direction, the line is removed and replaced by an extended Zig Zag line in the trend's original direction.

ZigZag Parameter Settings:

There are 3 parameters for input on the ZigZag indicator:

1\. Depth - with default value 12.

Depth – refers to how far back in the chart bar series it will look.

In order to get the highs and lows defined you need to make sure you have enough “Depth.”

It is the minimum number of bars without a second maximum or minimum deviation of the bar (example: if we have a maximum in candle x, and the depth is 12, it won't be able to draw the following maximum until at least x+12 candles).

2\. Deviation - with default value 5.

Deviation - refers to what percentage in price change it takes to change the trendline from positive to negative.

3\. Backstep - with default value 3.

Backstep - the minimum number of bars between swing highs and lows

> The ZigZag Indicator Formula:

> ZigZag (HL, %change=X, retrace=FALSE, LastExtreme=TRUE)
>
> If %change>=X,plot ZigZag

> where:
>
> HL = High-Low price series or Closing price series
>
> %change = Minimum price movement, in percentage
>
> Retrace = Is change a retracement of the previous move or an absolute change from peak to trough?
>
> Last Extreme = If the extreme price is the same over multiple periods, is the extreme price the first or last observation?

In my observations, for the ZigZag indicator there are at least 4 signal algorithms that can be used.

And in this expert advisor I created an option so that traders can choose and try the four algorithm signals from ZigZag indicator.

2.2. Awesome Oscillator (AO).

The Awesome Oscillator (AO) is a momentum indicator used by traders to identify market trends and momentum. It was developed by Bill Williams, a well-known technical analyst, and has become a popular tool among traders due to its simplicity and reliability.

The awesome oscillator is a market momentum indicator which compares recent market movements to historic market movements.

It uses a zero line in the centre, either side of which price movements are plotted according to a comparison of two different moving averages.

The Awesome Oscillator is calculated by subtracting a 34-period simple moving average (SMA) from a 5-period SMA of the midpoint (H+L)/2 price of a financial instrument.

The mid-point price is considered to be a more accurate representation of the true market price than either the open or close prices, as it takes into account both the high and low prices of a given period.

The AO oscillates between positive and negative values, with positive values indicating a bullish trend and negative values indicating a bearish trend.

The values of the 34-period simple moving average (SMA) and 5-period SMA are permanently set in the indicator code and are not given the opportunity to be changed in the indicator input parameter properties.

> Awesome Oscillator = 5-period SMA (median price, 5-periods) – 34-period SMA (median price, 34-periods)
>
> Where median price is: (High price of a session period + low price of a session period) / 2.

For the AO indicator there are at least 3 signal algorithms that can be used.

Therefore, in this expert advisor I created an option so that traders can choose and try the three algorithm signals from this AO indicator.

An illustration of the ZigZag indicator which are filtered with the Awesome Oscillator or filter each other's signals can be seen in Figure below.

The signal illustrations below are option number 2 ZigZag signal and option number 2 AO signal

[![ZZ_AO_Indi_Signal_illustration](https://c.mql5.com/2/70/ZZ_AO_Indi_Signal_illustration__1.png)](https://c.mql5.com/2/70/ZZ_AO_Indi_Signal_illustration.png "https://c.mql5.com/2/70/ZZ_AO_Indi_Signal_illustration.png")

![ZZ_AO_Indi_Signal_illustration_variation](https://c.mql5.com/2/70/ZZ_AO_Indi_Signal_illustration_variation.png)

**3\. Trade & Order Management**

There are several ways that provide manage trades with this multi-currency expert advisor:

3.1. Stop Loss Orders.

Options: Use Order Stop Loss (Yes) or (No)

- If the Use Order Stop Loss (No) option is selected, then all orders will be opened without a stop loss.

> Opening an order without a stop loss will be safe provided that Close Trade By Opposite Signal is set to Yes. But if the Close Trade By Opposite Signal is set to No, then there is a very high risk to equity. Therefore, I have added a function to enable checking the percentage between equity and balance. In this case, if the equity percentage of the balance is smaller than (100% - Percent Equity Risk per Trade) then the  expert will run the CheckLoss() function (a function for running a virtual stop loss) and will close orders that have a loss greater than set stop loss value.
>
> ```
>             //--
>             if(use_sl==No && CheckVSLTP==Yes)
>               {
>                 if(!mc.CheckEquityBalance())
>                   if(mc.CloseAllLoss())
>                     mc.Do_Alerts(symbol,"Close order due stop in loss to secure equity.");
>               }
>             //--
> ```

> ```
> bool MCEA::CheckEquityBalance(void)
>   {
> //---
>    bool isgood=false;
>    if((mc_account.Equity()/mc_account.Balance()*100) > (100.00-Risk)) isgood=true;
>    //--
>    return(isgood);
> //---
>   } //-end CheckEquityBalance()
> //---------//
> ```

> ```
> bool MCEA::CheckLoss(const string symbol,ENUM_POSITION_TYPE intype,double slc=0.0)
>    {
> //---
>      Pips(symbol);
>      bool inloss=false;
>      double lossval=slc==0.0 ? (SLval*0.5) : slc;
>      double posloss  = mc_symbol.NormalizePrice(slc*pip);
>      int ttlorder=PositionsTotal(); // number of open positions
>      //--
>      for(int x=0; x<arrsymbx; x++)
>        {
>          string symbol=DIRI[x];
>          //--
>          for(int i=ttlorder-1; i>=0; i--)
>             {
>               string position_Symbol   = PositionGetSymbol(i);
>               ENUM_POSITION_TYPE  type = mc_position.PositionType();
>               if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>                 {
>                   double price    = mc_position.PriceCurrent();
>                   double pos_open = mc_position.PriceOpen();
>                   double posloss  = mc_symbol.NormalizePrice(lossval*pip);
>                   double pricegab = mc_symbol.NormalizePrice(fabs(price-pos_open));
>                   //---
>                   if(type==intype && pricegab>posloss) inloss=true;
>                 }
>             }
>        }
>      //--
>      return(inloss);
> //----
>    } //-end CheckLoss()
> //---------//
> ```

> ```
> bool MCEA::CloseAllLoss(void)
>    {
> //----
>     ResetLastError();
>     //--
>     bool orclose=false;
>     string isloss="due stop in loss.";
>     //--
>     MqlTradeRequest req={};
>     MqlTradeResult  res={};
>     MqlTradeCheckResult check={};
>     //--
>     int ttlorder=PositionsTotal(); // number of open positions
>     //--
>     for(int x=0; x<arrsymbx; x++)
>        {
>          string symbol=DIRI[x];
>          Pips(symbol);
>          double posloss=mc_symbol.NormalizePrice(SLval*pip);
>          orclose=false;
>          //--
>          for(int i=ttlorder-1; i>=0; i--)
>             {
>               string position_Symbol   = PositionGetSymbol(i);
>               ENUM_POSITION_TYPE  type = mc_position.PositionType();
>               if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>                 {
>                   double price    = mc_position.PriceCurrent();
>                   double pos_open = mc_position.PriceOpen();
>                   double posloss  = mc_symbol.NormalizePrice(SLval*pip);
>                   double pricegab = mc_symbol.NormalizePrice(fabs(price-pos_open));
>                   ulong  position_ticket = PositionGetTicket(i);
>                   //---
>                   if(type==POSITION_TYPE_BUY && pricegab>posloss)
>                     {
>                       RefreshTick(position_Symbol);
>                       orclose = mc_trade.PositionClose(position_Symbol,slip);
>                       //--- output information about the closure
>                       PrintFormat("Close Buy %s %s %s",symbol,EnumToString(POSITION_TYPE_BUY),isloss);
>                     }
>                   if(type==POSITION_TYPE_SELL && pricegab>posloss)
>                     {
>                       RefreshTick(position_Symbol);
>                       orclose = mc_trade.PositionClose(position_Symbol,slip);
>                       //--- output information about the closure
>                       PrintFormat("Close Sell %s %s %s",symbol,EnumToString(POSITION_TYPE_BUY),isloss);
>                     }
>                 }
>             }
>        }
>      //--
>      return(orclose);
> //----
>    } //-end CloseAllLoss()
> //---------//
> ```

- If the option Use Order Stop Loss (Yes): Again given the option: Use Automatic Calculation Stop Loss(Yes) or (No)

- If the option Automatic Calculation Stop Loss (Yes), then the Stop Loss calculation will be performed automatically by the Expert.

- If the option Automatic Calculation Stop Loss (No), then the trader must Input Stop Loss value in Pips.

- If the option Use Order Stop Loss (No): Then the Expert will check for each order opened, whether the signal condition is still good and order may be maintained in a profit or condition the signal has weakened and the order needs to be closed to save profit or signal condition has reversed direction and order must be closed in a loss condition.

Note: Especially for Close Trade and Save profit due to weak signal, an option is given, whether to activate it or not.

> In part of the ExpertActionTrade() function:

> ```
>                 //--
>                 if(SaveOnRev==Yes) //-- Close Trade and Save profit due to weak signal (Yes)
>                   {
>                     mc.CheckOpenPMx(symbol);
>                     if(mc.profitb[x]>mc.minprofit && mc.xob[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Buy)==mc.Sell)
>                       {
>                         mc.CloseBuyPositions(symbol);
>                         mc.Do_Alerts(symbol,"Close BUY order "+symbol+" to save profit due to weak signal.");
>                       }
>                     if(mc.profits[x]>mc.minprofit && mc.xos[x]>0 && mc.GetCloseInWeakSignal(symbol,mc.Sell)==mc.Buy)
>                       {
>                         mc.CloseSellPositions(symbol);
>                         mc.Do_Alerts(symbol,"Close SELL order "+symbol+" to save profit due to weak signal.");
>                       }
>                   }
>                 //--
> ```

> The code to close the trade and save the profit due to a weak signal is as follows:
>
> ```
> int MCEA::GetCloseInWeakSignal(const string symbol,int exis) // Signal Indicator Position Close in profit
>   {
> //---
>     int ret=0;
>     int rise=1,
>         down=-1;
>     //--
>     int AOdir=AOColorSignal(symbol);
>     int ZZDir=ZigZagSignal(symbol);
>     //--
>     if(exis==down && (AOdir==rise||ZZDir==rise)) ret=rise;
>     if(exis==rise && (AOdir==down||ZZDir==down)) ret=down;
>     //--
>     return(ret);
> //---
>   } //-end GetCloseInWeakSignal()
> //---------//
> ```

- If it is not activated (No), even though the signal has weakened, the order will still be maintained or will not be closed to save profit.

- If activated (Yes), the conditions for the ZigZag indicator and AO indicator are:

> For close the Buy orders:
>
> When the AO indicator changes color from green to red (default color) or when the ZigZag indicator reaches an extreme high position and price changes direction, the Buy order will be closed.
>
> For close the Sell orders:
>
> When the AO indicator changes color from red to green (default color) or when the ZigZag indicator reaches an extreme low position and price changes direction, the Sell order will be closed.
>
> The code for setting a stop loss order is as follows:
>
> ```
> double MCEA::OrderSLSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice)
>   {
> //---
>     slv=0.0;
>     int x=PairsIdxArray(xsymb);
>     Pips(xsymb);
>     RefreshTick(xsymb);
>     //--
>     switch(type)
>       {
>        case (ORDER_TYPE_BUY):
>          {
>            if(use_sl==Yes && autosl==Yes) slv=mc_symbol.NormalizePrice(atprice-38*pip);
>            else
>            if(use_sl==Yes && autosl==No)  slv=mc_symbol.NormalizePrice(atprice-SLval*pip);
>            else slv=0.0;
>            //--
>            break;
>          }
>        case (ORDER_TYPE_SELL):
>          {
>            if(use_sl==Yes && autosl==Yes) slv=mc_symbol.NormalizePrice(atprice+38*pip);
>            else
>            if(use_sl==Yes && autosl==No)  slv=mc_symbol.NormalizePrice(atprice+SLval*pip);
>            else slv=0.0;
>          }
>       }
>     //---
>     return(slv);
> //---
>   } //-end OrderSLSet()
> //---------//
> ```

3.2. Take Profit orders.

Options: Use Order Take Profit (Yes) or (No)

- If the Use Order Take Profit (No) option is selected, then all orders will be opened without take profit.

- If the option Use Order Take Profit (Yes): Again given the option: Use Automatic Calculation Order Take Profit (Yes) or (No)

- If the option Automatic Calculation Order Take Profit (Yes), then the calculation of the Take Profit Order will be carried out automatically by the Expert.

- If the option Automatic Calculation Order Take Profit (No), then the trader must Input Order Take Profit value in Pips.

> The code to set a Take Profit order is as follows:
>
> ```
> double MCEA::OrderTPSet(const string xsymb,ENUM_ORDER_TYPE type,double atprice)
>   {
> //---
>     tpv=0.0;
>     int x=PairsIdxArray(xsymb);
>     Pips(xsymb);
>     RefreshTick(xsymb);
>     //--
>     switch(type)
>       {
>        case (ORDER_TYPE_BUY):
>          {
>            if(use_tp==Yes && autotp==Yes) tpv=mc_symbol.NormalizePrice(atprice+50*pip);
>            else
>            if(use_tp==Yes && autotp==No)  tpv=mc_symbol.NormalizePrice(atprice+TPval*pip);
>            else tpv=0.0;
>            //--
>            break;
>          }
>        case (ORDER_TYPE_SELL):
>          {
>            if(use_tp==Yes && autotp==Yes) tpv=mc_symbol.NormalizePrice(atprice-50*pip);
>            else
>            if(use_tp==Yes && autotp==No)  tpv=mc_symbol.NormalizePrice(atprice-TPval*pip);
>            else tpv=0.0;
>          }
>       }
>     //---
>     return(tpv);
> //---
>   } //-end OrderTPSet()
> //---------//
> ```

3.3. Trailing Stop

Options: Use Trailing Stop Loss (Yes) or (No)

- If the Use Trailing SL option is (No), then the Expert will not do trailing stop loss and trailing take profit.

- If the option Use Trailing SL (Yes): Traders can choose between 3 options:

> 1\. Trailing by Price
>
> The trailing stop will be performed by the Expert using price movements and the value in the input property.

> 2\. Trailing By Indicator
>
> The trailing stop will be executed by the Expert using the VIDYA indicator.
>
> According to my research and experiments, the VIDYA indicator is slightly better and ideal for trailing stops compared to the Parabolic SAR or several variants of Moving Average indicators.
>
> Compared to the Parabolic SAR indicator, the VIDYA indicator is closer to the price movements, and compared to the AMA, DEMA and MA indicators, the VIDYA indicator is even further away from the price movements.
>
> So in this article I decided to use the VIDYA indicator for the trailing stop function based on the indicator.

> 3\. Trailing Stop in HIGH or LOW previous bar
>
> For Buy orders, the trailing stop position will be placed at the LOW price of the previous bar (LOW\[1\]).
>
> For Sell orders, the trailing stop position will be placed at the HIGH price of the previous bar (HIGH\[1\]).

> Trailing Stop Price function:
>
> ```
> double MCEA::TSPrice(const string xsymb,ENUM_POSITION_TYPE ptype,int TS_type)
>   {
> //---
>     int br=2;
>     double pval=0.0;
>     int x=PairsIdxArray(xsymb);
>     Pips(xsymb);
>     //--
>     switch(TS_type)
>       {
>         case byprice:
>           {
>             RefreshTick(xsymb);
>             if(ptype==POSITION_TYPE_BUY)  pval=mc_symbol.NormalizePrice(mc_symbol.Bid()-TSval*pip);
>             if(ptype==POSITION_TYPE_SELL) pval=mc_symbol.NormalizePrice(mc_symbol.Ask()+TSval*pip);
>             break;
>           }
>         case byindi:
>           {
>             double VIDyAv[];
>             ArrayResize(VIDyAv,br,br);
>             ArraySetAsSeries(VIDyAv,true);
>             CopyBuffer(hVIDyAv[x],0,0,br,VIDyAv);
>             RefreshPrice(xsymb,TFt,br);
>             //--
>             if(ptype==POSITION_TYPE_BUY  && (VIDyAv[0]<mc_symbol.NormalizePrice(mc_symbol.Bid()-TSval*pip)))
>                pval=VIDyAv[0];
>             if(ptype==POSITION_TYPE_SELL && (VIDyAv[0]>mc_symbol.NormalizePrice(mc_symbol.Ask()+TSval*pip)))
>                pval=VIDyAv[0];
>             break;
>           }
>         case byHiLo:
>           {
>             UpdatePrice(xsymb,TFt,2);
>             //--
>             if(ptype==POSITION_TYPE_BUY  && (HIGH[0]>HIGH[1]))
>                pval=LOW[1];
>             if(ptype==POSITION_TYPE_SELL && (LOW[0]<LOW[1]))
>                pval=HIGH[1];
>             break;
>           }
>       }
>     //--
>     return(pval);
> //---
>   } //-end TSPrice()
> //---------//
> ```

> ```
> bool MCEA::ModifyOrdersSL(const string symbx,int TS_type)
>   {
> //---
>    ResetLastError();
>    MqlTradeRequest req={};
>    MqlTradeResult  res={};
>    MqlTradeCheckResult check={};
>    //--
>    int TRSP=TS_type;
>    bool modist=false;
>    int x=PairsIdxArray(symbx);
>    Pips(symbx);
>    //--
>    int total=PositionsTotal();
>    //--
>    for(int i=total-1; i>=0; i--)
>      {
>        string symbol=PositionGetSymbol(i);
>        if(symbol==symbx && mc_position.Magic()==magicEA)
>          {
>            ENUM_POSITION_TYPE opstype = mc_position.PositionType();
>            if(opstype==POSITION_TYPE_BUY)
>              {
>                RefreshTick(symbol);
>                double price = mc_position.PriceCurrent();
>                double vtrsb = mc_symbol.NormalizePrice(TSPrice(symbx,opstype,TRSP));
>                double pos_open   = mc_position.PriceOpen();
>                double pos_stop   = mc_position.StopLoss();
>                double pos_tp     = mc_position.TakeProfit();
>                double pos_profit = mc_position.Profit();
>                double pos_swap   = mc_position.Swap();
>                double pos_comm   = mc_position.Commission();
>                double netp=pos_profit+pos_swap+pos_comm;
>                double modstart=mc_symbol.NormalizePrice(pos_open+TSmin*pip);
>                double modminsl=mc_symbol.NormalizePrice(vtrsb+((TSmin-1.0)*pip));
>                double modbuysl=vtrsb;
>                bool modbuy = (price>modminsl && modbuysl>modstart && (pos_stop==0.0||modbuysl>pos_stop));
>                //--
>                if(modbuy && netp>minprofit)
>                  {
>                    modist=mc_trade.PositionModify(symbol,modbuysl,pos_tp);
>                  }
>              }
>            if(opstype==POSITION_TYPE_SELL)
>              {
>                RefreshTick(symbol);
>                double price = mc_position.PriceCurrent();
>                double vtrss = mc_symbol.NormalizePrice(TSPrice(symbx,opstype,TRSP));
>                double pos_open   = mc_position.PriceOpen();
>                double pos_stop   = mc_position.StopLoss();
>                double pos_tp     = mc_position.TakeProfit();
>                double pos_profit = mc_position.Profit();
>                double pos_swap   = mc_position.Swap();
>                double pos_comm   = mc_position.Commission();
>                double netp=pos_profit+pos_swap+pos_comm;
>                double modstart=mc_symbol.NormalizePrice(pos_open-TSmin*pip);
>                double modminsl=mc_symbol.NormalizePrice(vtrss-((TSmin+1.0)*pip));
>                double modselsl=vtrss;
>                bool modsel = (price<modminsl && modselsl<modstart && (pos_stop==0.0||modselsl<pos_stop));
>                //--
>                if(modsel && netp>minprofit)
>                  {
>                    modist=mc_trade.PositionModify(symbol,modselsl,pos_tp);
>                  }
>              }
>          }
>      }
>     //--
>     return(modist);
> //---
>   } //-end ModifyOrdersSL()
> //---------//
> ```

3.4. Trailing Take Profit

Options: Use Trailing Take Profit (Yes) or (No)

- If the Use Trailing TP option is (No), then the Expert will not do trailing take profit.

- If the Use Trailing TP option is (Yes), then Input Trailing Profit Value in Pips (default value 25 pips) and the expert advisor will perform trailing profit based on the variable value TPmin (minimum trailing profit value).

> ```
> bool MCEA::ModifyOrdersTP(const string symbx)
>   {
> //---
>    ResetLastError();
>    MqlTradeRequest req={};
>    MqlTradeResult  res={};
>    MqlTradeCheckResult check={};
>    //--
>    bool modist=false;
>    int x=PairsIdxArray(symbx);
>    Pips(symbx);
>    //--
>    int total=PositionsTotal();
>    //--
>    for(int i=total-1; i>=0; i--)
>      {
>        string symbol=PositionGetSymbol(i);
>        if(symbol==symbx && mc_position.Magic()==magicEA)
>          {
>            ENUM_POSITION_TYPE opstype = mc_position.PositionType();
>            if(opstype==POSITION_TYPE_BUY)
>              {
>                RefreshTick(symbol);
>                double price    = mc_position.PriceCurrent();
>                double pos_open = mc_position.PriceOpen();
>                double pos_stop = mc_position.StopLoss();
>                double pos_tp   = mc_position.TakeProfit();
>                double modbuytp = pos_tp==0.0 ? mc_symbol.NormalizePrice(pos_open+TPmin*pip) : pos_tp;
>                double modpostp = mc_symbol.NormalizePrice(price+TPmin*pip);
>                bool modtpb = (price>pos_open && modbuytp-price<TPmin*pip && pos_tp<modpostp);
>                //--
>                if(modtpb)
>                  {
>                    modist=mc_trade.PositionModify(symbol,pos_stop,modpostp);
>                  }
>              }
>            if(opstype==POSITION_TYPE_SELL)
>              {
>                RefreshTick(symbol);
>                double price    = mc_position.PriceCurrent();
>                double pos_open = mc_position.PriceOpen();
>                double pos_stop = mc_position.StopLoss();
>                double pos_tp   = mc_position.TakeProfit();
>                double modseltp = pos_tp==0.0 ? mc_symbol.NormalizePrice(pos_open-TPmin*pip) : pos_tp;
>                double modpostp = mc_symbol.NormalizePrice(price-TPmin*pip);
>                bool modtps = (price<pos_open && price-modseltp<TPmin*pip && pos_tp>modpostp);
>                //--
>                if(modtps)
>                  {
>                    modist=mc_trade.PositionModify(symbol,pos_stop,modpostp);
>                  }
>              }
>          }
>      }
>     //--
>     return(modist);
> //---
>   } //-end ModifyOrdersTP()
> //---------//
> ```

3.5. Close Trade By Opposite Signal

Options: Close Trade By Opposite Signal (Yes) or (No)

- If Close Trade By Opposite Signal (Yes):

> So, if a Sell order has previously been opened and then the indicator signal reverses, then the Sell order will be closed and then the expert advisor will open a Buy order. Likewise vice versa.

> In part of the ExpertActionTrade() function:

> ```
> if(Close_by_Opps==Yes && mc.xos[x]>0) mc.CloseSellPositions(symbol);
> ```
>
> ```
> bool MCEA::CloseSellPositions(const string symbol)
>   {
>     //---
>     ResetLastError();
>     bool sellclose=false;
>     int total=PositionsTotal(); // number of open positions
>     ENUM_POSITION_TYPE closetype = POSITION_TYPE_SELL;
>     ENUM_ORDER_TYPE     type_req = ORDER_TYPE_BUY;
>     //--
>     MqlTradeRequest req={};
>     MqlTradeResult  res={};
>     MqlTradeCheckResult check={};
>     //--
>     int x=PairsIdxArray(symbol);
>     //--- iterate over all open positions
>     for(int i=total-1; i>=0; i--)
>       {
>         if(mc_position.SelectByIndex(i))
>           {
>             //--- Parameters of the order
>             string position_Symbol   = PositionGetSymbol(i);
>             ulong  position_ticket   = PositionGetTicket(i);
>             ENUM_POSITION_TYPE  type = mc_position.PositionType();
>             //--- if the MagicNumber matches
>             if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>               {
>                 //--
>                 if(type==closetype)
>                   {
>                     RefreshTick(position_Symbol);
>                     sellclose=mc_trade.PositionClose(position_Symbol,slip);
>                     //--- output information about the closure
>                     PrintFormat("Close Sell #%I64d %s %s",position_ticket,position_Symbol,EnumToString(type));
>                   }
>               }
>           }
>       }
>    //---
>    return(sellclose);
> //----
>    } //-end CloseSellPositions()
> //---------//
> ```

> In part of the ExpertActionTrade() function:
>
> ```
> if(Close_by_Opps==Yes && mc.xob[x]>0) mc.CloseBuyPositions(symbol);
> ```

> ```
> bool MCEA::CloseBuyPositions(const string symbol)
>    {
>  //---
>     //--
>     ResetLastError();
>     bool buyclose=false;
>     int total=PositionsTotal(); // number of open positions
>     ENUM_POSITION_TYPE closetype = POSITION_TYPE_BUY;
>     ENUM_ORDER_TYPE     type_req = ORDER_TYPE_SELL;
>     //--
>     MqlTradeRequest req={};
>     MqlTradeResult  res={};
>     MqlTradeCheckResult check={};
>     //--
>     int x=PairsIdxArray(symbol);
>     //--- iterate over all open positions
>     for(int i=total-1; i>=0; i--)
>       {
>         if(mc_position.SelectByIndex(i))
>           {
>             //--- Parameters of the order
>             string position_Symbol   = PositionGetSymbol(i);
>             ulong  position_ticket   = PositionGetTicket(i);
>             ENUM_POSITION_TYPE  type = mc_position.PositionType();
>             //--- if the MagicNumber matches
>             if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>               {
>                 //--
>                 if(type==closetype)
>                   {
>                     RefreshTick(position_Symbol);
>                     buyclose=mc_trade.PositionClose(position_Symbol,slip);
>                     //--- output information about the closure
>                     PrintFormat("Close Buy #%I64d %s %s",position_ticket,position_Symbol,EnumToString(type));
>                   }
>               }
>           }
>       }
>    //---
>    return(buyclose);
> //----
>    } //-end CloseBuyPositions()
> //---------//
> ```

> Problems will arise if Close Trade By Opposite Signal (No):
>
> 1\. The order will be 2 times the total each pairs traded.
>
> 2\. In one pair there will be orders that are in a loss condition and there are also orders that are in a profit condition.
>
> 3\. The equity will be eroded due to unbalanced loss and profit conditions.
>
> To overcome this problem, I created several functions to detect loss and profit order conditions.
>
> In part of the ExpertActionTrade() function:
>
> ```
>             //--
>             mc.CheckOpenPMx(symbol);
>             if(Close_by_Opps==No && (mc.xob[x]+mc.xos[x]>1))
>               {
>                 if(mc.CheckProfitLoss(symbol))
>                    mc.Do_Alerts(symbol,"Close order due stop in loss.");
>               }
>             //--
> ```

> ```
> bool MCEA::CheckProfitLoss(const string symbol)
>    {
> //----
>      ResetLastError();
>      //--
>      bool closeinloss=false;
>      string isloss="due stop in loss.";
>      //--
>      int xx=PairsIdxArray(symbol);
>      //--
>      bool BuyProfitSellLoss=(xob[xx]>0 && CheckProfit(symbol,POSITION_TYPE_BUY)) && (xos[xx]>0 && CheckLoss(symbol,POSITION_TYPE_SELL,0.0));
>      bool SellProfitBuyLoss=(xos[xx]>0 && CheckProfit(symbol,POSITION_TYPE_SELL)) && (xob[xx]>0 && CheckLoss(symbol,POSITION_TYPE_BUY,0.0));
>      //--
>      if(BuyProfitSellLoss && !SellProfitBuyLoss)
>        {
>          if(CloseSellPositions(symbol))
>            {
>              PrintFormat("Close Sell %s %s %s",symbol,EnumToString(POSITION_TYPE_BUY),isloss);
>              closeinloss=true;
>            }
>        }
>      if(SellProfitBuyLoss && !BuyProfitSellLoss)
>        {
>          if(CloseBuyPositions(symbol))
>            {
>              PrintFormat("Close Buy %s %s %s",symbol,EnumToString(POSITION_TYPE_SELL),isloss);
>              closeinloss=true;
>            }
>        }
>      //--
>      return(closeinloss);
> //----
>    } //-end CheckProfitLoss()
> //---------//
> ```

> The CheckProfitLoss() function will call 2 other functions, which will compare the order in one pair with the conditions:
>
> Buy Profit and Sell Loss, or Buy Loss and Sell Profit.
>
> - If Buy Profit and Sell Loss, then the Sell order will be closed.
> - If Buy Loss and Sell Profit, then the Buy order will be closed.
>
> ```
> bool MCEA::CheckProfit(const string symbol,ENUM_POSITION_TYPE intype)
>    {
> //---
>      Pips(symbol);
>      double posprofit=mc_symbol.NormalizePrice((TPval*0.5)*pip);
>      bool inprofit=false;
>      //--
>      int ttlorder=PositionsTotal(); // number of open positions
>      //--
>      for(int x=0; x<arrsymbx; x++)
>        {
>          string symbol=DIRI[x];
>          //--
>          for(int i=ttlorder-1; i>=0; i--)
>             {
>               string position_Symbol   = PositionGetSymbol(i);
>               ENUM_POSITION_TYPE  type = mc_position.PositionType();
>               if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>                 {
>                   double price     = mc_position.PriceCurrent();
>                   double pos_open  = mc_position.PriceOpen();
>                   double posprofit = mc_symbol.NormalizePrice((TPval*0.5)*pip);
>                   double pricegab  = mc_symbol.NormalizePrice(fabs(price-pos_open));
>                   //---
>                   if(type==intype && posprofit<pricegab) inprofit=true;
>                 }
>             }
>        }
>      //--
>      return(inprofit);
> //----
>    } //-end CheckProfit()
> //---------//
> ```

> ```
> bool MCEA::CheckLoss(const string symbol,ENUM_POSITION_TYPE intype,double slc=0.0)
>    {
> //---
>      Pips(symbol);
>      bool inloss=false;
>      double lossval=slc==0.0 ? (SLval*0.5) : slc;
>      double posloss  = mc_symbol.NormalizePrice(slc*pip);
>      int ttlorder=PositionsTotal(); // number of open positions
>      //--
>      for(int x=0; x<arrsymbx; x++)
>        {
>          string symbol=DIRI[x];
>          //--
>          for(int i=ttlorder-1; i>=0; i--)
>             {
>               string position_Symbol   = PositionGetSymbol(i);
>               ENUM_POSITION_TYPE  type = mc_position.PositionType();
>               if((position_Symbol==symbol) && (mc_position.Magic()==magicEA))
>                 {
>                   double price    = mc_position.PriceCurrent();
>                   double pos_open = mc_position.PriceOpen();
>                   double posloss  = mc_symbol.NormalizePrice(lossval*pip);
>                   double pricegab = mc_symbol.NormalizePrice(fabs(price-pos_open));
>                   //---
>                   if(type==intype && pricegab>posloss) inloss=true;
>                 }
>             }
>        }
>      //--
>      return(inloss);
> //----
>    } //-end CheckLoss()
> //---------//
> ```

**4\. Manual Order Management.**

In this multi-currency expert advisor, several manual button clicks are added to provide efficiency and effectiveness for traders in monitoring the expert advisor's work.

> 4.1. Set SL / TP All Orders:
>
> This button is useful if the trader has entered the parameter sets Use Order Stop Loss (No) and/or Use Order Take Profit (No), but then the trader wants to use Stop Loss or Take Profit on all orders, then with a single click on the button "Set SL/TP All Orders" all orders will be modified and a Stop Loss and/or Take Profit will be applied.
>
> 4.2. Close All Orders:
>
> If a trader wishes to close all orders, a single click on the "Close All Orders" button will close all open orders.

> 4.3. Close All Orders Profit:
>
> If a trader wants to close all orders that are already profitable, a single click on the "Close All Orders Profit" button will close all open orders that are already profitable.

**5\. Management Orders and Chart Symbols.**

For multi-currency expert advisors who will trade 30 pairs from only one chart symbol, it would be very helpful and easy if provided a button panel for all symbols so that traders can change charts timeframe or symbols with just one click to see the accuracy of the indicator signal when the expert opens or closes an order.

### Implementation of planning in the MQL5 program

**1\. Program header and input properties.**

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

In the TFUSE enumeration, I limit the use of the time frame calculations for the experts only from TF-M5 to TF-D1.

Enumeration to select the type to be used in the Trailing Stop calculation

```
//--
enum TrType
  {
    byprice, // Trailing Stop by Price
    byindi,  // Trailing Stop by Indicator
    byHiLo   // Trailing Stop in HIGH or LOW bar
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

Enumeration to select the signal algorithm from the ZigZag indicator

```
//--
enum SignalZZ
 {
   SZZ1,  // ZigZagSignal 1
   SZZ2,  // ZigZagSignal 2
   SZZ3,  // ZigZagSignal 3
   SZZ4   // ZigZagSignal 4
 };
//--
```

Enumeration to select the signal algorithm from the AO indicator

```
//--
enum SignalAO
 {
   SAO1,  // AOSignal 1
   SAO2,  // AOSignal 2
   SAO3   // AOSignal 3
 };
//--
```

Expert input properties

```
//---
input group               "=== Global Strategy EA Parameter ==="; // Global Strategy EA Parameter
input TFUSE               tfinuse = TFH4;             // Select Expert TimeFrame, default PERIOD_H4
//---
input group               "=== ZigZag Indicator Input Properties ===";  // ZigZag Indicator Input Properties
input int                 zzDepth = 12;               // Input ZigZag Depth, default 12
input int                 zzDevia = 5;                // Input ZigZag Deviation, default 5
input int                 zzBackS = 3;                // Input ZigZag Back Step, default 3
input SignalZZ              sigzz = SZZ2;             // Select ZigZag Signal to Use
input SignalAO              sigao = SAO2;             // Select AO Signal to Use
//---
input group               "=== Selected Pairs to Trade ===";  // Selected Pairs to trading
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
input YN              TrailingSL = Yes;              // Use Trailing Stop Loss (Yes) or (No)
input TrType               trlby = byHiLo;           // Select Trailing Stop Type
input double               TSval = 10.0;             // If Use Trailing Stop by Price Input value in Pips
input double               TSmin = 5.0;              // Minimum Pips to start Trailing Stop
input YN              TrailingTP = Yes;              // Use Trailing Take Profit (Yes) or (No)
input double               TPmin = 25.0;             // Input Trailing Profit Value in Pips
input YN           Close_by_Opps = Yes;              // Close Trade By Opposite Signal (Yes) or (No)
input YN               SaveOnRev = Yes;              // Close Trade and Save profit due to weak signal (Yes) or (No)
input YN              CheckVSLTP = Yes;              // Check Virtual SL/TP & Close Loss Trade (Yes) or (No)
//--Others Expert Advisor Parameter
input group               "=== Others Expert Advisor Parameter ==="; // Others EA Parameter
input YN                  alerts = Yes;              // Display Alerts / Messages (Yes) or (No)
input YN           UseEmailAlert = No;               // Email Alert (Yes) or (No)
input YN           UseSendnotify = No;               // Send Notification (Yes) or (No)
input YN      trade_info_display = Yes;              // Select Display Trading Info on Chart (Yes) or (No)
input ulong              magicEA = 20240218;         // Expert ID (Magic Number)
//---
```

Note: If the input property of Expert ID (Magic Number) is left blank, the Expert Advisor will be able to manage orders opened manually.

In the Global Strategy EA Parameters expert input property group, traders are instructed to select the Expert Timeframe for indicator signal calculations with default PERIOD\_H4.

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

Note: The expert validates the pairs.  If the trader makes a mistake when entering the pair name (typos) or if the pair validation fails, the expert will receive a warning and the expert advisor will be removed from the chart.

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

**2\. Class for working Expert Advisor**

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
    int              GetIndiSignals(const string symbol);
    int              ZigZagSignal(const string symbol);
    int              AOSignal(const string symbol);
    int              AOColorSignal(const string symbol);
    int              PARSAR05(const string symbol);
    int              PARSAR15(const string symbol);
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

    //-- ZigZag_AO_MCEA Config --
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
    string           indiname1;
    //--
    //--- Indicators Handle
    int              hZigZag[],
                     hAO[];
    int              hVIDyAv[];
    int              hPar05[],
                     hPar15[];
    //---
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
    virtual void     ZigZag_AO_MCEA_Config(void);
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
    void             UpdatePrice(const string symbol,ENUM_TIMEFRAMES xtf,int bars);
    void             RefreshPrice(const string symbx,ENUM_TIMEFRAMES xtf,int bars);
    //--
    bool             CheckEquityBalance(void);
    bool             RefreshTick(const string symbx);
    bool             TradingToday(void);
    bool             OpenBuy(const string symbol);
    bool             OpenSell(const string symbol);
    bool             ModifyOrderSLTP(double mStop,double ordtp);
    bool             ModifyOrdersSL(const string symbx,int TS_type);
    bool             ModifyOrdersTP(const string symbx);
    bool             CloseAllProfit(void);
    bool             CloseAllLoss(void);
    bool             ManualCloseAllProfit(void);
    bool             CheckProfitLoss(const string symbol);
    bool             CloseBuyPositions(const string symbol);
    bool             CloseSellPositions(const string symbol);
    bool             CheckProfit(const string symbol,ENUM_POSITION_TYPE intype);
    bool             CheckLoss(const string symbol,ENUM_POSITION_TYPE intype,double slc=0.0);
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

The first and most important function in the Multi-Currency Expert Advisor workflow process is called from the OnInit() is ZigZag\_AO\_MCEA\_Config().

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
//---
   mc.ZigZag_AO_MCEA_Config();
   //--
   return(INIT_SUCCEEDED);
//---
  } //-end OnInit()
//---------//
```

The ZigZag\_AO\_MCEA\_Config() function configures all symbols to be used, all timeframes, all handle indicators and some important functions of the include file header for the Expert Advisor workflow.

The ZigZag\_AO\_MCEA\_Config() function describes and implements how to handle timeframes and create indicator handles for all indicators used in the Expert Advisor workflow.

```
//+------------------------------------------------------------------+
//| Expert Configuration                                             |
//+------------------------------------------------------------------+
void MCEA::ZigZag_AO_MCEA_Config(void)
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
        hZigZag[x] = iCustom(DIRI[x],TFt,indiname1,zzDepth,zzDevia,zzBackS);   //-- Handle for the ZigZag indicator
        hAO[x]     = iAO(DIRI[x],TFt);                                         //-- Handle for the Awesome_Oscillator indicator
        hVIDyAv[x] = iVIDyA(DIRI[x],TFt,9,12,0,PRICE_WEIGHTED);                //-- Handle for the VIDYA indicator for Trailing Stop
        hPar05[x]  = iSAR(DIRI[x],TFT05,SARstep,SARmaxi);                      //-- Handle for the iSAR indicator for M5 Timeframe
        hPar15[x]  = iSAR(DIRI[x],TFT15,SARstep,SARmaxi);                      //-- Handle for the iSAR indicator for M15 Timeframe
        //--
      }
    //--
    TesterHideIndicators(true);
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
  } //-end ZigZag_AO_MCEA_Config()
//---------//
```

**3\. Expert tick function and workflow**

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

The whole process of EA working for trading is included in this function.

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
                          mc.CloseAllProfit();
                        //--
                        if(mc.xos[x]>0 && use_sl==No && CheckVSLTP==Yes)
                          {
                            if(mc.CheckLoss(symbol,POSITION_TYPE_SELL,SLval))
                              if(mc.CloseSellPositions(symbol))
                                mc.Do_Alerts(symbol,"Check Profit Trade and Close order due stop in loss.");
                          }
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
                          mc.CloseAllProfit();
                        //--
                        if(mc.xob[x]>0 && use_sl==No && CheckVSLTP==Yes)
                          {
                            if(mc.CheckLoss(symbol,POSITION_TYPE_BUY,SLval))
                              if(mc.CloseBuyPositions(symbol))
                                mc.Do_Alerts(symbol,"Check Profit Trade and Close order due stop in loss.");
                          }
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
                if(TrailingSL==Yes) mc.ModifyOrdersSL(symbol,trlby); //-- Use Trailing Stop Loss (Yes)
                if(TrailingTP==Yes) mc.ModifyOrdersTP(symbol);       //-- Use Trailing Take Profit (Yes)
              }
            //--
            mc.CheckOpenPMx(symbol);
            if(Close_by_Opps==No && (mc.xob[x]+mc.xos[x]>1))
              {
                if(mc.CheckProfitLoss(symbol))
                   mc.Do_Alerts(symbol,"Close order due stop in loss.");
              }
            //--
            if(use_sl==No && CheckVSLTP==Yes)
              {
                if(!mc.CheckEquityBalance())
                  if(mc.CloseAllLoss())
                    mc.Do_Alerts(symbol,"Close order due stop in loss to secure equity.");
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

Notes: Day Trading On/Off conditions will be displayed in the Trading Info on Chart.

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

Notes: As explained above, in the case of trading on the New Zealand Session to trading on the US New York Session, the time from the start of trading to the end of trading is calculated by the EA.

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

**4\. How to get trading signals for open positions?**

In order to get a signal to open a position, the ExpertActionTrade() function calls the GetOpenPosition() function.

```
int MCEA::GetOpenPosition(const string symbol) // Signal Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    int ZZAOSignal=GetIndiSignals(symbol);
    int dirmove=DirectionMove(symbol,TFt);
    int psar15=PARSAR15(symbol);
    //--
    if(ZZAOSignal==rise && dirmove==rise && psar15==rise) ret=rise;
    if(ZZAOSignal==down && dirmove==down && psar15==down) ret=down;
    //--
    return(ret);
//---
  } //-end GetOpenPosition()
//---------//
```

And the GetOpenPosition() function will call 3 functions:

1. GetIndiSignals(symbol)
2. DirectionMove(symbol,TFt)
3. PARSAR15(symbol)

4.1. GetIndiSignals(symbol) function will call 2 functions:

1. ZigZagSignal(symbol)
2. AOSignal(symbol)

4.1.1. ZigZag Signal

Inside the ZigZagSignal() function, we use and call 1 function, which is thePairsIdxArray() function.

```
int xx=PairsIdxArray(symbol)
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

The PairsIdxArray() function is used to get the name of the requested symbol and the handles of its indicators.

Then the corresponding indicator handle is called to get the buffer value of the ZigZag indicator from that timeframe.

```
    //-- Indicators handle for all symbol
    for(int x=0; x<arrsymbx; x++)
      {
        hZigZag[x] = iCustom(DIRI[x],TFt,indiname1,zzDepth,zzDevia,zzBackS);   //-- Handle for the ZigZag indicator
        hAO[x]     = iAO(DIRI[x],TFt);                                         //-- Handle for the Awesome_Oscillator indicator
        hVIDyAv[x] = iVIDyA(DIRI[x],TFt,9,12,0,PRICE_WEIGHTED);                //-- Handle for the VIDYA indicator for Trailing Stop
        hPar05[x]  = iSAR(DIRI[x],TFT05,SARstep,SARmaxi);                      //-- Handle for the iSAR indicator for M5 Timeframe
        hPar15[x]  = iSAR(DIRI[x],TFT15,SARstep,SARmaxi);                      //-- Handle for the iSAR indicator for M15 Timeframe
        //--
      }
    //--
```

So, to get the buffer value of ZigZag indicator, we will copy each buffer from the ZigZag indicators handle.

To copy the ZigZag buffer (buffer 0) from the ZigZag indicator handle to the destination array:

```
CopyBuffer(hZigZag[x],0,0,barcalc,ZZBuffer);
```

Apart from that, it will also call the UpdatePrice() function to get the High price value and Low price value which will be used to get the ZigZag buffer High and ZigZag buffer Low bars positions.

```
void MCEA::UpdatePrice(const string symbol,ENUM_TIMEFRAMES xtf)
  {
//---
    //--
    ArrayFree(OPEN);
    ArrayFree(HIGH);
    ArrayFree(LOW);
    ArrayFree(CLOSE);
    ArrayFree(TIME);
    //--
    ArrayResize(OPEN,arper,arper);
    ArrayResize(HIGH,arper,arper);
    ArrayResize(LOW,arper,arper);
    ArrayResize(CLOSE,arper,arper);
    ArrayResize(TIME,arper,arper);
    //--
    ArraySetAsSeries(OPEN,true);
    ArraySetAsSeries(HIGH,true);
    ArraySetAsSeries(LOW,true);
    ArraySetAsSeries(CLOSE,true);
    ArraySetAsSeries(TIME,true);
    //--
    ArrayInitialize(OPEN,0.0);
    ArrayInitialize(HIGH,0.0);
    ArrayInitialize(LOW,0.0);
    ArrayInitialize(CLOSE,0.0);
    ArrayInitialize(TIME,0);
    //--
    RefreshPrice(symbol,xtf,arper);
    //--
    int co=CopyOpen(symbol,xtf,0,arper,OPEN);
    int ch=CopyHigh(symbol,xtf,0,arper,HIGH);
    int cl=CopyLow(symbol,xtf,0,arper,LOW);
    int cc=CopyClose(symbol,xtf,0,arper,CLOSE);
    int ct=CopyTime(symbol,xtf,0,arper,TIME);
   //--
   return;
//---
  } //-end UpdatePrice()
//---------//
```

And to get the bar positions of ZigZag High and ZigZag Low, we make iterations and compare with Price High and Price Low.

```
    //--
    for(int i=barcalc-1; i>=0; i--)
      {
        if(ZZBuffer[i]==HIGH[i]) ZH=i;
        if(ZZBuffer[i]==LOW[i])  ZL=i;
      }
    //--
```

After getting the bar positions of ZigZag High (ZH) and ZigZag Low (ZL), then it just depends on the signal ZigZag indicator option selected in the property input.

The complete ZigZagSignal() function is as follow:

```
int MCEA::ZigZagSignal(const string symbol) // ZigZag Signal for Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int ZH=-1,
        ZL=-1;
    int barcalc=100;
    bool ZZrise=false;
    bool ZZdown=false;
    //--
    double ZZBuffer[];
    ArrayResize(ZZBuffer,barcalc,barcalc);
    ArraySetAsSeries(ZZBuffer,true);
    //--
    int x=PairsIdxArray(symbol);
    UpdatePrice(symbol,TFt);
    //--
    CopyBuffer(hZigZag[x],0,0,barcalc,ZZBuffer);
    //--
    for(int i=barcalc-1; i>=0; i--)
      {
        if(ZZBuffer[i]==HIGH[i]) ZH=i;
        if(ZZBuffer[i]==LOW[i])  ZL=i;
      }
    //--
    switch(sigzz)
      {
        case SZZ1:
          {
            ZZrise=((ZH==0 && HIGH[0]>HIGH[1])||(ZL<ZH && ZL>1));
            ZZdown=((ZL==0 && LOW[0]<LOW[1])||(ZH<ZL && ZH>1));
            //--
            break;
          }
        case SZZ2:
          {
            ZZrise=(ZL<ZH && ZL>1);
            ZZdown=(ZH<ZL && ZH>1);
            //--
            break;
          }
        case SZZ3:
          {
            ZZrise=((ZH==0 && HIGH[0]>HIGH[1])||(ZL<ZH && ZL>0));
            ZZdown=((ZL==0 && LOW[0]<LOW[1])||(ZH<ZL && ZH>0));
            //--
            break;
          }
        case SZZ4:
          {
            ZZrise=(ZL<ZH && ZL>0);
            ZZdown=(ZH<ZL && ZH>0);
            //--
            break;
          }
      };
    //--
    if(ZZrise) ret=rise;
    if(ZZdown) ret=down;
    //--
    return(ret);
//---
  } //-end ZigZagSignal()
//---------//
```

4.1.2. AO Signal

Just like in the ZigZagSignal() function, in the AOSignal() function we also have to use thePairsIdxArray() function, to get the buffer value from the AO indicator.

So, to get the buffer value of AO indicator, we will copy each buffer from the AO indicators handle.

To copy the AO buffer (buffer 0) from the AO indicator handle to the destination array:

```
CopyBuffer(hAO[x],0,0,barcalc,AOValue);
```

And then it just depends on the signal AO indicator option selected in the property input.

Apart from that, to complete the signal from the AO indicator, we also use the color of the AO indicator to verify the signal from the buffer value.

And for that purpose, we created a function to get the color value of the AO indicator with the function name AOColorSignal()

In this function we will copy buffer 1 (indicator color index buffer) from the AO indicator.

```
CopyBuffer(hAO[x],1,0,barcalc,AOColor);
```

```
int MCEA::AOColorSignal(const string symbol)
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int barcalc=9;
    //--
    double AOColor[];
    ArrayResize(AOColor,barcalc,barcalc);
    ArraySetAsSeries(AOColor,true);
    //--
    int x=PairsIdxArray(symbol);
    UpdatePrice(symbol,TFt,barcalc);
    //--
    CopyBuffer(hAO[x],1,0,barcalc,AOColor);
    //--
    bool AORise=((AOColor[1]==1.0 && AOColor[0]==0.0)||(AOColor[1]==0.0 && AOColor[0]==0.0));
    bool AODown=((AOColor[1]==0.0 && AOColor[0]==1.0)||(AOColor[1]==1.0 && AOColor[0]==1.0));
    //--
    if(AORise) ret=rise;
    if(AODown) ret=down;
    //--
    return(ret);
//---
  } //-end AOColorSignal()
//---------//
```

The complete AOSignal() function is as follow:

```
int MCEA::AOSignal(const string symbol)
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int barcalc=9;
    bool AORValue=false;
    bool AODValue=false;
    //--
    double AOValue[];
    ArrayResize(AOValue,barcalc,barcalc);
    ArraySetAsSeries(AOValue,true);
    //--
    int x=PairsIdxArray(symbol);
    UpdatePrice(symbol,TFt,barcalc);
    //--
    CopyBuffer(hAO[x],0,0,barcalc,AOValue);
    //--
    switch(sigao)
      {
        case SAO1:
          {
            AORValue=(AOValue[2]<=0.0 && AOValue[1]>0.0 && AOValue[0]>AOValue[1])||(AOValue[1]>AOValue[2] && AOValue[0]>AOValue[1]);
            AODValue=(AOValue[2]>=0.0 && AOValue[1]<0.0 && AOValue[0]<AOValue[1])||(AOValue[1]<AOValue[2] && AOValue[0]<AOValue[1]);
            //--
            break;
          }
        case SAO2:
          {
            AORValue=(AOValue[1]<=0.0 && AOValue[0]>0.0)||(AOValue[0]>0.0 && AOValue[0]>AOValue[1]);
            AODValue=(AOValue[1]>=0.0 && AOValue[0]<0.0)||(AOValue[0]<0.0 && AOValue[0]<AOValue[1]);
            //--
            break;
          }
        case SAO3:
          {
            AORValue=(AOValue[1]<=0.0 && AOValue[0]>0.0)||(AOValue[0]>AOValue[1]);
            AODValue=(AOValue[1]>=0.0 && AOValue[0]<0.0)||(AOValue[0]<AOValue[1]);
            //--
            break;
          }
      };
    //--
    bool AORise=(AOColorSignal(symbol)==rise);
    bool AODown=(AOColorSignal(symbol)==down);
    //--
    if(AORValue && AORise) ret=rise;
    if(AODValue && AODown) ret=down;
    //--
    return(ret);
//---
  } //-end AOSignal()
//---------//
```

The GetIndiSignals() function will add up the return values from the ZigZagSignal() function and the AOSignal() function.

```
int MCEA::GetIndiSignals(const string symbol) // Get Signal for Open Position
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    int sigrise=2;
    int sigdown=-2;
    //--
    int ZZSignal=ZigZagSignal(symbol);
    int AwSignal=AOSignal(symbol);
    //Print(symbol+" = ZZ="+string(ZZSignal)+" AO="+string(AwSignal)+" Signal="+string(ZZSignal+AwSignal));
    //--
    if(ZZSignal+AwSignal==sigrise) ret=rise;
    if(ZZSignal+AwSignal==sigdown) ret=down;
    //--
    return(ret);
//---
  } //-end GetIndiSignals()
//---------//
```

The GetIndiSignals() function will add up the return values from the ZigZagSignal() function and the AOSignal function.

- If the result is 2, it is a signal to Buy.
- If the result is -2, it is a signal to Sell.

4.2. DirectionMove function.

The DirectionMove() function is useful for getting the close price position on the current bar, whether above the open price (up) or below the open price (down).

```
int MCEA::DirectionMove(const string symbol,const ENUM_TIMEFRAMES stf) // Bar Price Direction
  {
//---
    int ret=0;
    int rise=1,
        down=-1;
    //--
    Pips(symbol);
    double difud=mc_symbol.NormalizePrice(1.5*pip);
    UpdatePrice(symbol,stf,2);
    //--
    if(CLOSE[0]>OPEN[0]+difud) ret=rise;
    if(CLOSE[0]<OPEN[0]-difud) ret=down;
    //--
    return(ret);
//---
  } //-end DirectionMove()
//---------//
```

4.3. PARSAR15() function.

The PARSAR15() function is useful for aligning the movement of the ZigZag indicator and AO indicator with the Parabolic Stop and Reverse system indicator (PSAR/iSAR) on the M15 timeframe.

```
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
   UpdatePrice(symbol,TFT15,br);
   //--
   if(PSAR[0]<LOW[0])
      ret=rise;
   if(PSAR[0]>HIGH[0])
      ret=down;
//--
   return(ret);
//---
  } //-end PARSAR15()
//---------//
```

After executing the 3 main signal functions and several supporting functions, the GetOpenPosition() function will provide the values:

- Value = 0, signal unknown.
- Value = 1, is a signal for open Buy order;
- Value = -1, is a signal for open Sell order.

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

**5\. ChartEvent Function**

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

Then the Multi-Currency Expert Advisor ZigZag\_AO\_MCEA interface resembles the following figure.

![ZZ_AO_MCEA_Look](https://c.mql5.com/2/70/ZZ_AO_MCEA_Look.png)

As you can see, there are buttons "M", "C" and "R" under the Expert Advisor name ZigZag\_AO\_MCEA.

When the M button is clicked, a panel of manual clicking buttons is displayed as shown below.

> > > > > ![Expert_manual_button_01](https://c.mql5.com/2/70/Expert_manual_button_01.png)

The trader can manage orders manually when the manual click button panel is displayed:

5.1. Set SL/TP All Orders

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

5.2. Close All Orders

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

5.3. Close All Profits

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

Furthermore, when the C button is clicked, a panel button with 30 symbol names or pairs is displayed and traders can click on one of the pair names or symbol names.

Clicking on one of the pair names or symbol names immediately replaces the chart symbol with the symbol whose name was clicked.

> > > > > > > > ![Expert_manual_button_02](https://c.mql5.com/2/70/Expert_manual_button_02.png)

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

And finally, clicking on the R button will remove the Multi-Currency Expert Advisor ZigZag\_AO\_MCEA from the chart, so traders don't have to remove the experts manually.

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

Therefore, on the MetaTrader 5 Strategy Tester platform we will test the ZigZag\_AO\_MCEA Multi-Currency Expert Advisor.

In the test, we have placed the ZigZag\_AO\_MCEA on the XAUUSD pair and the H4 timeframe, with a custom time period of 2023.10.01 to 2024.02.17.

![ZZ_AO_ST_input](https://c.mql5.com/2/70/ZZ_AO_ST_input.png)

And the results are as in the following figures.

![ZZ_AO_ST_result](https://c.mql5.com/2/70/ZZ_AO_ST_result.png)

![ZZ_AO_ST_result_graph_01](https://c.mql5.com/2/73/ZZ_AO_ST_result_graph_01.png)

![ZZ_AO_ST_result_graph_02](https://c.mql5.com/2/73/ZZ_AO_ST_result_graph_02.png)

![ZZ_AO_ST_result_graph_03](https://c.mql5.com/2/73/ZZ_AO_ST_result_graph_03.png)

![ZZ_AO_ST_result_graph_04](https://c.mql5.com/2/73/ZZ_AO_ST_result_graph_04.png)

### Conclusion

The conclusion in creating a Multi-Currency Expert Advisor with signals from ZigZag indicator which are filtered with the Awesome Oscillator or filter each other's signals for forex trading using MQL5 is as follows:

1. It turns out that creating a multi-currency Expert Advisor in MQL5 is very simple and not much different from creating a single-currency Expert Advisor, where the multi-currency Expert Advisor can also be used as a single-currency Expert Advisor.
2. Creating a Multi-Currency Expert Advisor will increase the efficiency and effectiveness of traders by eliminating the need to open many chart symbols for trading.
3. Applying the right trading strategy will increase the probability of profit compared to using a single-currency Expert Advisor. This is because losses in one pair will be covered by profits in other pairs.
4. This ZigZag\_AO\_MCEA Multi-Currency Expert Advisor is just a sample for learning and idea generation. The test results on the Strategy Tester are still not good. Therefore, by experimenting and testing on different timeframes or different indicator period calculations, and different signal selected it is possible to get better strategy and more profitable results.
5. In my opinion, this ZigZag with AO indicators strategy should be further researched with various different experiments, starting from timeframe, differentiation value of the ZigZag indicator parameter input, and maybe you can also add other algorithm signals for the ZigZag indicator and AO indicator.
6. Based on the results of my experiments on Strategy Tester, on timeframes below H1 the results are not good, only on timeframes H1 and above, the results are good with few open trades, compared to small timeframes with many open trades, but in loss.

We hope that this article and the MQL5 Multi-Currency Expert Advisor program will be useful for traders to learn and develop ideas.

Thanks for reading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14329.zip "Download all attachments in the single ZIP archive")

[ZigZag\_AO\_MCEA.mq5](https://www.mql5.com/en/articles/download/14329/zigzag_ao_mcea.mq5 "Download ZigZag_AO_MCEA.mq5")(123.74 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 5): Bollinger Bands On Keltner Channel — Indicators Signal](https://www.mql5.com/en/articles/13861)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 4): Triangular moving average — Indicator Signals](https://www.mql5.com/en/articles/13770)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 3): Added symbols prefixes and/or suffixes and Trading Time Session](https://www.mql5.com/en/articles/13705)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 2): Indicator Signals: Multi Timeframe Parabolic SAR Indicator](https://www.mql5.com/en/articles/13470)
- [How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 1): Indicator Signals based on ADX in combination with Parabolic SAR](https://www.mql5.com/en/articles/13008)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/464039)**
(3)


![Soshianth Azar](https://c.mql5.com/avatar/2021/8/610B070F-9A56.jpg)

**[Soshianth Azar](https://www.mql5.com/en/users/alakialakiani)**
\|
1 Apr 2024 at 12:33

Dear  [Roberto Jacobs](https://www.mql5.com/en/users/3rjfx "3rjfx")

Hello to you and everyone

Thank you so much for your great article.

I ran the codes in the strategy tester and in several brokers and unfortunately every time I get the error 'order's limit reached'. I will be very grateful if you can help me to solve this problem.

With respect and best wishes

[![](https://c.mql5.com/3/432/Screenshot_2024-04-01_140221__1.png)](https://c.mql5.com/3/432/Screenshot_2024-04-01_140221.png "https://c.mql5.com/3/432/Screenshot_2024-04-01_140221.png")

![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
1 Apr 2024 at 14:06

**Soshianth Azar [#](https://www.mql5.com/en/forum/464039#comment_52899497):**

Dear  [Roberto Jacobs](https://www.mql5.com/en/users/3rjfx "3rjfx")

Hello to you and everyone

Thank you so much for your great article.

I ran the codes in the strategy tester and in several brokers and unfortunately every time I get the error 'order's limit reached'. I will be very grateful if you can help me to solve this problem.

With respect and best wishes

Please ask your broker about the AccountInfoInteger(ACCOUNT\_LIMIT\_ORDERS) of your account, or whether your account is currently limited.

![germor](https://c.mql5.com/avatar/avatar_na2.png)

**[germor](https://www.mql5.com/en/users/germor)**
\|
13 Sep 2024 at 21:12

Hi Roberto!

Interesting article. May I know if you have tested it on a demo or [real account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties")? And what results did you get?

![MQL5 Wizard Techniques you should know (Part 13): DBSCAN for Expert Signal Class](https://c.mql5.com/2/73/MQL5_Wizard_Techniques_you_should_know_Part_13_DBSCAN_for_Expert_Signal_Class___LOGO.png)[MQL5 Wizard Techniques you should know (Part 13): DBSCAN for Expert Signal Class](https://www.mql5.com/en/articles/14489)

Density Based Spatial Clustering for Applications with Noise is an unsupervised form of grouping data that hardly requires any input parameters, save for just 2, which when compared to other approaches like k-means, is a boon. We delve into how this could be constructive for testing and eventually trading with Wizard assembled Expert Advisers

![Developing an MQTT client for Metatrader 5: a TDD approach — Part 6](https://c.mql5.com/2/73/Developing_an_MQTT_client_for_Metatrader_5_PArt_6____LOGO.png)[Developing an MQTT client for Metatrader 5: a TDD approach — Part 6](https://www.mql5.com/en/articles/14391)

This article is the sixth part of a series describing our development steps of a native MQL5 client for the MQTT 5.0 protocol. In this part we comment on the main changes in our first refactoring, how we arrived at a viable blueprint for our packet-building classes, how we are building PUBLISH and PUBACK packets, and the semantics behind the PUBACK Reason Codes.

![Trader-friendly stop loss and take profit](https://c.mql5.com/2/60/Trader_friendly_stop_loss_and_take_profit_LOGO.png)[Trader-friendly stop loss and take profit](https://www.mql5.com/en/articles/13737)

Stop loss and take profit can have a significant impact on trading results. In this article, we will look at several ways to find optimal stop order values.

![Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://c.mql5.com/2/60/Neural_networks_are_easy_wPart_636_Logo.png)[Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://www.mql5.com/en/articles/13712)

We continue to discuss the family of Decision Transformer methods. From previous article, we have already noticed that training the transformer underlying the architecture of these methods is a rather complex task and requires a large labeled dataset for training. In this article we will look at an algorithm for using unlabeled trajectories for preliminary model training.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/14329&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5048821115784109904)

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