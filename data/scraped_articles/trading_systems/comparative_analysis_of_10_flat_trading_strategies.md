---
title: Comparative analysis of 10 flat trading strategies
url: https://www.mql5.com/en/articles/4534
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:53:05.922366
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/4534&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083184277241206416)

MetaTrader 5 / Trading systems


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/4534#intro)
- [Defining the task when creating a flat trading strategy](https://www.mql5.com/en/articles/4534#task)
- [Flat trading strategies](https://www.mql5.com/en/articles/4534#strategies)

  - [Strategy #1. The Envelopes indicator with the MFI-based filter](https://www.mql5.com/en/articles/4534#str1)
  - [Strategy #2. Bollinger Bands and two Moving Averages](https://www.mql5.com/en/articles/4534#str2)
  - [Strategy #3. WSO & WRO Channel with a filter based on Fractal Dimension Ehlers](https://www.mql5.com/en/articles/4534#str3)
  - [Strategy #4. The Percentage Crossover Channel indicator and the TrendRange-based filter](https://www.mql5.com/en/articles/4534#str4)
  - [Strategy #5. The Price Channel indicator and the RBVI-based filter](https://www.mql5.com/en/articles/4534#str5)
  - [Strategy #6. The Williams Percent Range indicator and the ADX-based filter](https://www.mql5.com/en/articles/4534#str6)
  - [Strategy #7. Modified Keltner channel and the Magic Trend-based filter](https://www.mql5.com/en/articles/4534#str7)
  - [Strategy #8. Donchian channel with a confirmation by Trinity Impulse](https://www.mql5.com/en/articles/4534#str8)
  - [Strategy #9. The ATR Channel indicator and a filter based on CCI Color Levels](https://www.mql5.com/en/articles/4534#str9)
  - [Strategy #10. RSI histogram and a filter based on the Flat indicator](https://www.mql5.com/en/articles/4534#str10)

- [Testing](https://www.mql5.com/en/articles/4534#tests)
- [Findings](https://www.mql5.com/en/articles/4534#conclusions)
- [Conclusion](https://www.mql5.com/en/articles/4534#final)

### Introduction

Trend following strategies are very popular and easy-to-use, especially for beginners. However, current markets have become more dynamic, while trend movements are less distinct (in terms of both range and duration). By not using the possibility of trading in flat or sideways markets, we lose potential profits. Trend following trading rules are simple: identify the signs of a trend and try to capitalize on it. Trading in flat periods differs much from that. During sideways movement, the price is in a small range and may stay unchanged for quite a long time. There is no directional movement in the market, and liquidity is low.

### Defining the task when creating a flat trading strategy

[In the](https://www.mql5.com/en/articles/4534#final) [previous article](https://www.mql5.com/en/articles/3074#task), I defined three tasks, which were required for creating a trend following strategy. Tasks required for the creation of flat trading strategies are very similar.

![](https://c.mql5.com/2/32/001__1.jpg)

Fig. 1. Example of a sideways/flat movement.

_Task 1. Identifying the presence of a flat period._

There is no general and exhaustive definition of a flat (actually there is no proper description of the concept of a trend). However, there can be certain indications of that the market is currently in a flat state. This movement is also called sideways, because there is no clear vertical movement both upwards and downwards. The price moves inside a range, approaching its lower and upper borders in waves. Another sign of a flat period can be low volume of trades in the market or low interest from market participants. This can be seen from a weak price change, as well as from the small tick volume.

_Task 2. The targets of an open position._

Channel trading is often used in relation to flat trading techniques. This is the major method of using the sideways movement for the purpose of profiting. A flat channel is determined in some virtual borders. Further, the trading strategy is built based on relations between the price and the channel borders. Most often, the strategy implies buying or selling when the price bounces off the channel border (Fig. 2).

![](https://c.mql5.com/2/33/1-2.jpg)

Fig. 2. Trading when price bounces off channel borders.

When selling in the upper part of the channel, we assume that the price will move towards the lower border. That will act as a take profit level. Stop loss can be set as a certain numerical value in points or in accordance with the channel border. A counter strategy is used for Buy operations: buying at the lower channel border and setting the take profit level near the upper border.

### Flat trading strategies

I used the above principles when choosing flat trading strategies.

- Trading will be performed inside the channel. Therefore, we need to choose tools that will help us build a channel and determine the virtual borders of the flat zone.
- In addition to defining the channel, we need at least one more additional tool to confirm that the price will go in the right direction after rebounding from the channel border. The purpose of such a filter is to avoid false entry signals.


#### Strategy \#1. The Envelopes indicator with the MFI-based filter

The channel borders are determined based on the Envelopes indicator. The MFI indicator is additionally used for filtering of signals.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Envelopes](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/envelopes") |
| Used indicator | [MFI](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/market_facilitation "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/market_facilitation") |
| Timeframe | H1 |
| Buy conditions | The price reaches the lower channel border, and MFI is in the oversold zone (below 20) |
| Sell conditions | The price reaches the upper channel border, and MFI is in the overbought zone (above 80) |
| Exit conditions | The price reaches the opposite channel border |

Fig. 3 shows market entry conditions according to Strategy #1.

![](https://c.mql5.com/2/32/1.gif)

Fig. 3. Entry conditions for the flat trading Strategy #1

The code of the Expert Advisor trading this strategy is shown below:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;
      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(mfi[0]<20 && env_low[0]>close[0])
     {
      tp=env_high[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(mfi[0]>80 && env_high[0]<close[0])
     {
      tp=env_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,mfi)<=0  ||
          CopyBuffer(InpInd_Handle2,1,0,2,env_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,env_high)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

Take Profit is set automatically in accordance with the set conditions, while Stop Loss is set manually depending on the timeframe.

#### Strategy \#2. Bollinger Bands and two Moving Averages

Channel borders are determined using Bollinger Bands, and signals are filtered based on the relative position of the slow and fast MAs.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Bollinger Bands](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/bb") |
| Used indicator | [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") |
| Timeframe | H1 |
| Buy conditions | The price reaches the lower channel border, the fast MA is above the slow one |
| Sell conditions | The price reaches the upper channel border, the fast MA is below the slow one |
| Exit conditions | The price reaches the opposite channel border |

Fig. 4 shows market entry conditions. The default periods of two SMAs are small: 4 and 8. Period values and smoothing methods are adjustable, so you can change the filtering sensitivity for Bollinger Bands signals.

![](https://c.mql5.com/2/32/2.gif)

Fig. 4. Entry conditions for the flat trading Strategy #2

Except for market entry conditions, strategy #2 is very similar to strategy #1.

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;
      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(ma_slow[0]>ma_fast[0] && bb_low[0]>close[0])
     {
      tp=bb_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(ma_slow[0]<ma_fast[0] && bb_up[0]<close[0])
     {
      tp=bb_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,1,0,2,bb_up)<=0  ||
          CopyBuffer(InpInd_Handle1,2,0,2,bb_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,ma_slow)<=0 ||
          CopyBuffer(InpInd_Handle3,0,0,2,ma_fast)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#3. WSO & WRO Channel and Fractal Dimension Ehlers

The main signal indicator is WSO & WRO Channel, which is a channel based on two oscillators: WSO (Widner Support Oscillator) and WRO (Widner Resistance Oscillator). The idea of the indicator is based on the article "Automated Support and Resistance" by Mel Widner. To filter the signals, we will use the Fractal Dimension indicator, which was described in the article "Fractal Dimension as a Market Mode Sensor" by John F. Ehlers and Ric Way.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [WSO & WRO Channel](https://www.mql5.com/en/code/20589) |
| Used indicator | [Fractal Dimension Ehlers](https://www.mql5.com/en/code/20580) |
| Timeframe | Any |
| Buy conditions | The price reaches the lower channel border, and the Fractal Dimension value is below the threshold |
| Sell conditions | The price reaches the upper channel border, and the Fractal Dimension value is below the threshold |
| Exit conditions | The price reaches the opposite channel border |

Fig. 5 shows market entry conditions. Similar to previous strategies, trading operations are performed when the price bounces off the channel borders. The filter helps find the entry points when the market is not trending.

![](https://c.mql5.com/2/32/3__1.gif)

Fig. 5. Entry conditions for the flat trading Strategy #3

The code of the Expert Advisor trading this strategy is shown below:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;
      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(wwc_low[0]>close[0] && fdi[0]<Inp_FdiThreshold)
     {
      tp=wwc_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(wwc_up[0]<close[0] && fdi[0]<Inp_FdiThreshold)
     {
      tp=wwc_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,3,0,2,wwc_up)<=0  ||
          CopyBuffer(InpInd_Handle1,2,0,2,wwc_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,fdi)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#4. The Percentage Crossover Channel indicator and the TrendRange-based filter

In this case, the channel is built at the breakout of levels by a certain percentage. We need an indicator building the channel, search for points where the price bounces off the channel borders and a signals filter. We will use the TrendRange indicator, which shows both trend and flat states. The states will be applied for filtering signals.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Percentage Crossover Channel](https://www.mql5.com/en/code/19796) |
| Used indicator | [Trend Range](https://www.mql5.com/en/code/20156) |
| Timeframe | Any |
| Buy conditions | The price reaches the lower channel border, and the Trend Range histogram is gray |
| Sell conditions | The price reaches the upper channel border, and the Trend Range histogram is gray |
| Exit conditions | The price reaches the opposite channel border |

Market entry conditions are shown in fig.6. The Percentage Crossover indicator has a number of specific features. The Percent parameter is the limit distance, after breaking which a new level construction begins, and the parameter is timeframe dependent. A smaller percent should be set on lower timeframes. For example, the recommended value for the hourly timeframe is 20 — 30. Higher values ​​result in excessive selectivity of the indicator.

![](https://c.mql5.com/2/32/004.gif)

Fig. 6. Entry conditions for the flat trading Strategy #4

The strategy code is shown below:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;
      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(pcc_low[0]>close[0] && tr_flat[0]>tr_range[0])
     {
      tp=pcc_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(pcc_up[0]<close[0] && tr_flat[0]>tr_range[0])
     {
      tp=pcc_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,pcc_up)<=0  ||
          CopyBuffer(InpInd_Handle1,2,0,2,pcc_low)<=0 ||
          CopyBuffer(InpInd_Handle2,1,0,2,tr_flat)<=0 ||
          CopyBuffer(InpInd_Handle2,2,0,2,tr_range)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#5. The Price Channel indicator and the RBVI-based filter

The Price Channel indicator builds the channel, whose upper and lower borders are determined by the highest and the lowest prices over the period. False signals will be filtered out by RBVI, which determines the presence of a flat period in the market.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Price Channel](https://www.mql5.com/en/code/44) |
| Used indicator | [RBVI](https://www.mql5.com/en/code/719) |
| Timeframe | Any |
| Buy conditions | The price reaches the lower channel border, and the RBVI value is below the threshold |
| Sell conditions | The price reaches the upper channel border, and the RBVI value is below the threshold |
| Exit conditions | The price reaches the opposite channel border |

Entry conditions are shown in fig.7. RBVI threshold value is 40. The value can be changed in the Expert Advisor parameters.

![](https://c.mql5.com/2/32/005__2.gif)

Fig. 7. Entry conditions for the flat trading Strategy #5

Here is the code of the strategy:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(pc_low[0]>close[0] && rbvi[0]<=Inp_level)
     {
      tp=pc_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(pc_up[0]<close[0] && rbvi[0]<=Inp_level)
     {
      tp=pc_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,pc_up)<=0  ||
          CopyBuffer(InpInd_Handle1,1,0,2,pc_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,rbvi)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#6. The Williams Percent Range indicator and the ADX-based filter

Williams's Percent Range determining the overbought/oversold state is used for finding entry points. Since our purpose is to trade in flat period or when the price is supposed to return to a certain range, let's use the trend indicator ADX in order to determine the absence of a directional movement.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Williams Percent Range](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/wpr") |
| Used indicator | [ADX](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/admi "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/admi") |
| Timeframe | Any |
| Buy conditions | The WPR indicator is in the oversold zone (below -80) and the ADX value is below the threshold. |
| Sell conditions | The WPR indicator is in the overbought zone (above -20) and the ADX value is below the threshold. |
| Exit conditions | Take Profit/Stop Loss |

As can be seen in Fig. 8, the default flat area by ADX is set to 30. The value can be customized in the Expert Advisor code.

![](https://c.mql5.com/2/32/6.gif)

Fig. 8. Entry conditions for the flat trading Strategy #6

The strategy implementation is provided in the below listing. Here the Inp\_FlatLevel variable sets the above mentioned ADX threshold value.

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,Inp_StopLoss,Inp_TakeProfit,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,Inp_StopLoss,Inp_TakeProfit,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   return(wpr[0]<-80 && adx[0]<Inp_FlatLevel)?true:false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   return(wpr[0]>=-20 && adx[0]<Inp_FlatLevel)?true:false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,wpr)<=0  ||
          CopyBuffer(InpInd_Handle2,0,0,2,adx)<=0)?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#7. Modified Keltner channel and the Magic Trend-based filter

Price's bouncing off the Keltner channel is checked using the Magic Trend indicator, which should identify flat zones.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Modified Keltner Channel](https://www.mql5.com/en/code/19825) |
| Used indicator | [Magic Trend](https://www.mql5.com/en/code/20337) |
| Timeframe | Any |
| Buy conditions | The price reaches the lower channel border and the Magic Trend line is gray |
| Sell conditions | The price reaches the upper channel border and the Magic Trend line is gray |
| Exit conditions | The price reaches the opposite channel border |

This trading strategy is visualized in figure 9. The Magic Trend value is not changed during flat, i.e. it is shown as a gray horizontal line. Hence, in addition to price's reaching the channel border, we check the condition of Magic Trend, which should be in a flat state for some time. This check will be implemented as a comparison of values ​​on the current bar and the previous one - they should be the same.

![](https://c.mql5.com/2/32/7.gif)

Fig. 9. Entry conditions for the flat trading Strategy #7.

The code contains functions checking Buy/Sell conditions:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(kc_low[0]>close[0] && mt[0]==mt[1])
     {
      tp=kc_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(kc_up[0]<close[0] && mt[0]==mt[1])
     {
      tp=kc_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,kc_up)<=0  ||
          CopyBuffer(InpInd_Handle1,2,0,2,kc_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,mt)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#8. Donchian channel with a confirmation by Trinity Impulse

In this case, we try to catch the moments when the price bounces off the Donchian channel borders, while the Trinity Impulse indicator is in the sideways movement state.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [Donchian Channel](https://www.mql5.com/en/code/1601) |
| Used indicator | [Trinity Impulse](https://www.mql5.com/en/code/684) |
| Timeframe | Lower timeframes |
| Buy conditions | The price reaches the lower channel border, the Trinity Impulse value is zero |
| Sell conditions | The price reaches the upper channel border, the Trinity Impulse value is zero |
| Exit conditions | The price reaches the opposite channel border |

Market entries are shown in figure 10. The strategy is not recommended for use on higher timeframes, because the Trinity Impulse filter shows jig saw behavior, while the display of flat zones is lagging. Their width is very small, and this makes the strategy too selective. Optimally use 5 to 30-minute timeframes.

![](https://c.mql5.com/2/32/8.gif)

Fig. 10. Entry conditions for the flat trading Strategy #8.

Implementation of the Expert Advisor based on the above strategy:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(dc_low[0]>close[0] && ti[0]==0)
     {
      tp=dc_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(dc_up[0]<close[0] && ti[0]==0)
     {
      tp=dc_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,dc_up)<=0  ||
          CopyBuffer(InpInd_Handle1,1,0,2,dc_low)<=0 ||
          CopyBuffer(InpInd_Handle2,0,0,2,ti)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#9. The ATR Channel indicator and a filter based on CCI Color Levels

ATR Channel is based on ATR deviations from a moving average. CCI Color Levels is a CCI indicator displayed as a histogram of threshold values, which indicate price movement. We use this indicator to filter the channel's flat state (when CCI is between the threshold values).

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [ATR Channel](https://www.mql5.com/en/code/766) |
| Used indicator | [CCI Color Levels](https://www.mql5.com/en/code/19704) |
| Timeframe | Any |
| Buy conditions | The price reaches the lower channel border, and the CCI Color Levels value is in the range between the thresholds |
| Sell conditions | The price reaches the upper channel border, and the CCI Color Levels value is in the range between the thresholds |
| Exit conditions | The price reaches the opposite channel border |

Fig. 11 shows the market entry. In some cases, the price may exit the channel, but the CCI-based filter in the specified range suggests that the price may return to the channel and reach the set Take Profit value.

![](https://c.mql5.com/2/32/9__1.gif)

Fig. 11. Entry conditions for the flat trading Strategy #9.

The code of the Expert Advisor trading this strategy:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,close[0]-Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,close[0]+Inp_StopLoss*_Point,tp,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   if(atr_low[0]>close[0] && cci[0]<Inp_CCI_LevelUP && cci[0]>Inp_CCI_LevelDOWN)
     {
      tp=atr_up[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   if(atr_up[0]<close[0] && cci[0]<Inp_CCI_LevelUP && cci[0]>Inp_CCI_LevelDOWN)
     {
      tp=atr_low[0];
      return true;
     }
   else
      return false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,1,0,2,atr_up)<=0  ||
          CopyBuffer(InpInd_Handle1,2,0,2,atr_low)<=0 ||
          CopyBuffer(InpInd_Handle2,2,0,2,cci)<=0 ||
          CopyClose(Symbol(),PERIOD_CURRENT,0,2,close)<=0
          )?false:true;
  }
//+------------------------------------------------------------------+
```

#### Strategy \#10. RSI histogram and a filter based on the Flat indicator

RSI in the form of a histogram provides a better visualization, since the main market entry signals are generated in its overbought/oversold zones. Flat is used to filter out false signals.

| Indicator Parameter | Description |
| --- | --- |
| Used indicator | [RSI\_Histogram](https://www.mql5.com/en/code/14923) |
| Used indicator | [Flat](https://www.mql5.com/en/code/2060) |
| Timeframe | Any |
| Buy conditions | RSI is in the oversold zone (below the threshold) and Flat is in the flat zone. |
| Sell conditions | RSI is in the overbought zone (above the threshold) and Flat is in the flat zone. |
| Exit conditions | Take Profit/Stop Loss |

Entry points are shown in figure 12. The histogram form of RSI provides for an easier tracking of overbought/oversold zones and of the Flat filter's flat zone.

![](https://c.mql5.com/2/32/10.gif)

Fig. 12.Entry conditions for the flat trading Strategy #10.

Implementation of the Expert Advisor based on the above strategy is available in the following code:

```
void OnTick()
  {
//--- Checking orders previously opened by the EA
   if(!Trade.IsOpenedByMagic(Inp_MagicNum))
     {
      //--- Getting data for calculations
      if(!GetIndValue())
         return;

      //--- Opening an order if there is a buy signal
      if(BuySignal())
         Trade.BuyPositionOpen(Symbol(),Inp_Lot,Inp_StopLoss,Inp_TakeProfit,Inp_MagicNum,Inp_EaComment);

      //--- Opening an order if there is a sell signal
      if(SellSignal())
         Trade.SellPositionOpen(Symbol(),Inp_Lot,Inp_StopLoss,Inp_TakeProfit,Inp_MagicNum,Inp_EaComment);
     }
  }
//+------------------------------------------------------------------+
//| Buy conditions                                                   |
//+------------------------------------------------------------------+
bool BuySignal()
  {
   return(rsi[0]<Inp_LowLevel && fl[0]<Inp_FLowLevel)?true:false;
  }
//+------------------------------------------------------------------+
//| Sell conditions                                                  |
//+------------------------------------------------------------------+
bool SellSignal()
  {
   return(rsi[0]>Inp_HighLevel && fl[0]<Inp_FLowLevel)?true:false;
  }
//+------------------------------------------------------------------+
//| Getting the current values of indicators                         |
//+------------------------------------------------------------------+
bool GetIndValue()
  {
   return(CopyBuffer(InpInd_Handle1,0,0,2,rsi)<=0  ||
          CopyBuffer(InpInd_Handle2,0,0,2,fl)<=0)?false:true;
  }
//+------------------------------------------------------------------+
```

### Testing

Now, that we have defined 10 flat trading strategies and implemented them in the code, let's select common testing conditions.

- Testing interval: **Last year**.
- Currency pair: **EURUSD**.
- Trading mode: **No delay (** These are not high-frequency trading strategies, so the effect of delays would be very small).

- Testing: **M1 OHLC**(Pre-testing on real ticks shows nearly the same results).

- Initial deposit: **1000 USD.**
- Leverage: **1:500.**
- Server: **MetaQuotes-Demo.**
- Quotes: **5-digit.**

**Strategy #1 Test (the Envelopes indicator with the MFI based filter)**

Preset:

```
input int                  Inp_StopLoss=500;                            //Stop Loss(points)

//--- MFI indicator parameters
input ENUM_APPLIED_VOLUME  Inp_applied_volume=VOLUME_TICK;              // MFI volume type
input int                  Inp_MFI_period=10;                           // MFI period
//--- Envelopes indicator parameters
input int                  Inp_ma_period=10;                            // Envelopes MA period
input ENUM_MA_METHOD       Inp_ma_method=MODE_SMA;                      // Envelopes smoothing type
input double               Inp_deviation=0.1;                           // deviation of borders from the Envelopes MA
```

Testing results:

![](https://c.mql5.com/2/32/001.gif)

Fig. 13. Flat Strategy #1. Testing results.

**Strategy #2 Test (Bollinger Bands and two Moving Averages)**

Preset:

```
input int                  Inp_StopLoss=450;                            //Stop Loss(points)

//--- Bollinger Bands parameters
input int                  Inp_BBPeriod=14;                             //BB period
input double               Inp_deviation=2.0;                           //Deviation
//--- MA slow parameters
input int                  Inp_ma_period1=12;                           //MA slow period
input ENUM_MA_METHOD       Inp_ma_method1=MODE_SMMA;                    //MA slow smoothing method
//--- MA fast parameters
input int                  Inp_ma_period2=2;                            //MA fast period
input ENUM_MA_METHOD       Inp_ma_method2=MODE_LWMA;                    //MA fast smoothing method
```

Testing results:

![](https://c.mql5.com/2/32/002.gif)

Fig. 14. Flat Strategy #2. Testing results.

**Strategy #3 Test (WSO & WRO Channel with a filter based on Fractal Dimension Ehlers)**

Preset:

```
input int                  Inp_StopLoss=500;                            //Stop Loss(points)

//--- WSO & WRO Channel parameters
input int                  Inp_WsoWroPeriod=16;                         //Wso & Wro Channel period
//--- Fractal Dimension Ehlers parameters
input int                  Inp_FdiPeriod    =  18;                      //Fractal dimension period
input double               Inp_FdiThreshold =  1.4;                     //Fractal dimension threshold
input ENUM_APPLIED_PRICE   Inp_Price        = PRICE_CLOSE;              //Applied price
```

Testing results:

![](https://c.mql5.com/2/32/003.gif)

Fig. 15. Flat Strategy #3. Testing results.

**Strategy #4 Test (the Percentage Crossover Channel indicator and the TrendRange based filter)**

Preset:

```
input int                  Inp_StopLoss=500;                            //Stop Loss(points)

//--- Percentage_Crossover_Channel parameters
input double               Inp_Percent=26.0;                            //Percentage of the limit distance
input ENUM_APPLIED_PRICE   Inp_Price=PRICE_CLOSE;                       //Applied price
//--- Trend Range indicator parameters
input uint                 Inp_PeriodTR    =  14;                       //Trend Range period
input ENUM_MA_METHOD       Inp_Method      =  MODE_EMA;                 //Smoothing method
input double               Inp_Deviation   =  1.0;                      //Deviation
```

Testing results:

![](https://c.mql5.com/2/32/004__1.gif)

Fig.16. Flat Strategy #4. Testing results.

**Strategy #5 Test (the Price Channel indicator and the RBVI based filter)**

Preset:

```
input int                  Inp_StopLoss=450;                            //Stop Loss(points)

//--- Price Channel indicator parameters
input int                  Inp_ChannelPeriod=12;                        //Period
//--- RBVI indicator parameters
input int                  Inp_RBVIPeriod=5;                            //RBVI period
input ENUM_APPLIED_VOLUME  Inp_VolumeType=VOLUME_TICK;                  //volume
input double               Inp_level=40;                                //flat level
```

Testing results:

![](https://c.mql5.com/2/32/005__3.gif)

Fig. 17. Flat Strategy #5. Testing results.

**Strategy #6 Test (the Williams Percent Range indicator and the ADX based filter)**

Preset:

```
input int                  Inp_StopLoss=50;                             //Stop Loss(points)
input int                  Inp_TakeProfit=50;                           //Take Profit(points)

//--- WPR indicator parameters
input int                  Inp_WPRPeriod=10;                            //Period of WPR
//--- ADX indicator parameter
input int                  Inp_ADXPeriod=14;                            //Period of ADX
input int                  Inp_FlatLevel=40;                            //Flat Level of ADX
```

Testing results:

![](https://c.mql5.com/2/32/006__2.gif)

Fig. 18. Flat Strategy #6. Testing results.

**Strategy #7 Test (Modified Keltner channel and the Magic Trend based filter)**

Preset:

```
input int                     Inp_SmoothCenter      =  11;              // Number of the periods to smooth the center line
input int                     Inp_SmoothDeviation   =  12;              // Number of periods to smooth deviation
input double                  Inp_F                 =  1.0;             // Factor which is used to apply the deviation
input ENUM_APPLIED_PRICE      Inp_AppliedPrice      =  PRICE_CLOSE;     // The center line applied price:
input ENUM_MA_METHOD          Inp_MethodSmoothing   =  MODE_SMA;        // The center line smoothing method
input ENUM_METHOD_VARIATION   Inp_MethodVariation   =  METHOD_HL;       // Variation Method
//--- Magic Trend indicator parameters
input uint                    Inp_PeriodCCI   =  60;                    // CCI period
input uint                    Inp_PeriodATR   =  5;                     // ATR period
```

Testing results:

![](https://c.mql5.com/2/32/007__2.gif)

Fig. 19. Flat Strategy #7. Testing results.

**Strategy #8 Test (Donchian channel with a confirmation by Trinity Impulse)**

Preset:

```
input int                  Inp_StopLoss=500;                            //Stop Loss(points)

//--- Donchian channel parameters
input int                  Inp_ChannelPeriod=12;                        //Donchian period
//--- Trinity Impulse indicator parameters
input int                  Inp_Period= 5;                               //Indicator period
input int                  Inp_Level= 34;                               //Smoothing level
input ENUM_MA_METHOD       Inp_Type=MODE_LWMA;                          //Averaging type
input ENUM_APPLIED_PRICE   Inp_Price=PRICE_WEIGHTED;                    //Price
input ENUM_APPLIED_VOLUME  Inp_Volume=VOLUME_TICK;                      //Volume type
```

Testing results:

![](https://c.mql5.com/2/32/008__2.gif)

Fig. 20. Flat Strategy #8. Testing results.

**Strategy #9 Test (the ATR Channel indicator and a filter based on CCI Color Levels)**

Preset:

```
//--- ATR Channel parameters
input ENUM_MA_METHOD       Inp_MA_Method=MODE_SMA;                      //MA smoothing method
input uint                 Inp_MA_Period=10;                            //MA period
input uint                 Inp_ATR_Period=12;                           //ATR period
input double               Inp_Factor=1.5;                              //Number of deviations
input ENUM_APPLIED_PRICE   Inp_IPC=PRICE_LOW;                           //Applied price
input int                  Inp_Shift=0;                                 //Horizontal shift of the indicator in bars
//--- CCI Color Levels parameters
input int                  Inp_CCI_ma_period = 14;                      // Averaging period
input double               Inp_CCI_LevelUP   = 90;                      // Level UP
input double               Inp_CCI_LevelDOWN =-90;                      // Level DOWN
```

Testing results:

![](https://c.mql5.com/2/32/009__2.gif)

Fig. 21. Flat Strategy #9. Testing results.

**Strategy #10 Test (RSI histogram and a filter based on the Flat indicator)**

Preset:

```
//--- RSI Histogram parameters
input uint                 Inp_RSIPeriod=12;                            // indicator period
input ENUM_APPLIED_PRICE   Inp_RSIPrice=PRICE_CLOSE;                    // price
input uint                 Inp_HighLevel=60;                            // overbought level
input uint                 Inp_LowLevel=40;                             // oversold level
input int                  Inp_Shift=0;                                 // horizontal shift of the indicator in bars
//--- Flat indicator parameters
input uint                 Inp_Smooth=10;                               // Smoothing period
input ENUM_MA_METHOD       Inp_ma_method=MODE_SMA;                      // Smoothing type
input ENUM_APPLIED_PRICE   Inp_applied_price=PRICE_CLOSE;               // Price type
input uint                 Inp_HLRef=100;
input int                  Inp_FShift=0;                                // Horizontal shift of the indicator in bars
input uint                 Inp_ExtraHighLevel=70;                       // Maximum trend level
input uint                 Inp_FHighLevel=50;                           // Strong trend level
input uint                 Inp_FLowLevel=30;                            // Weak trend level
```

Testing results:

![](https://c.mql5.com/2/32/010__2.gif)

Fig. 22. Flat Strategy #10. Testing results.

### Findings

Testing and optimization of the analyzed strategies for trading the sideways market have produced the following results.

- Most of the strategies are based on trading inside a channel and include signal filtering, so their main weak point is a short-term channel breakout.
- Testing on very low and very high timeframes resulted in losses due to frequent market entry conditions and due to an extreme selectivity, respectively.
- No significant deviations in returns were obtained during optimization on the same currency pair and period of time. Almost all strategies showed similar results.

Thus, we can draw the main conclusion: although we selected different channel building techniques and filters, the advantages and disadvantages of all strategies are comparable.

### Conclusion

Below is a summary table of the names of Expert Advisors, which were developed and used in this article, as well as auxiliary classes and a list of indicators used in the above strategies. The archive attached below contains all described files properly arranged into folders. For their proper operation, you only need to save the **MQL5** folder into the terminal folder.

**Programs used in the article:**

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Strategy\_1.mq5 | Expert Advisor | Strategy #1. The Envelopes indicator with the MFI-based filter |
| 2 | Strategy\_2.mql5 | Expert Advisor | Strategy #2. Bollinger Bands and two Moving Averages |
| 3 | Strategy\_3.mq5 | Expert Advisor | Strategy #3. WSO & WRO Channel with a filter based on Fractal Dimension Ehlers |
| 4 | Strategy\_4.mq5 | Expert Advisor | Strategy #4. The Percentage Crossover Channel indicator and the TrendRange-based filter |
| 5 | Strategy\_5.mq5 | Expert Advisor | Strategy #5. The Price Channel indicator and the RBVI-based filter |
| 6 | Strategy\_6.mq5 | Expert Advisor | Strategy #6. The Williams Percent Range indicator and the ADX-based filter |
| 7 | Strategy\_7.mq5 | Expert Advisor | Strategy #7. Modified Keltner channel and the Magic Trend-based filter |
| 8 | Strategy\_8.mq5 | Expert Advisor | Strategy #8. Donchian channel with a confirmation by Trinity Impulse |
| 9 | Strategy\_9.mq5 | Expert Advisor | Strategy #9. The ATR Channel indicator and a filter based on CCI Color Levels |
| 10 | Strategy\_10.mq5 | Expert Advisor | Strategy #10. RSI histogram and a filter based on the Flat indicator |
| 11 | Trade.mqh | Library | A class of trading functions |
| 13 | wwc.mq5 | Indicator | Used in Strategy #3 |
| 14 | fdi.mq5 | Indicator | Used in Strategy #3 |
| 15 | pcc.mq5 | Indicator | Used in Strategy #4 |
| 16 | trend\_range.mq5 | Indicator | Used in Strategy #4 |
| 17 | price\_channel.mq5 | Indicator | Used in Strategy #5 |
| 18 | rbvi.mq5 | Indicator | Used in Strategy #5 |
| 19 | customizable \_keltner.mq5 | Indicator | Used in Strategy #7 |
| 20 | magic\_trend.mq5 | Indicator | Used in Strategy #7 |
| 21 | donchian\_channel.mq5 | Indicator | Used in Strategy #8 |
| 22 | trinity\_impulse.mq5 | Indicator | Used in Strategy #8 |
| 23 | atr\_channel.mq5 | Indicator | Used in Strategy #9 |
| 24 | cci\_color\_levels.mq5 | Indicator | Used in Strategy #9 |
| 25 | rsi\_histogram.mq5 | Indicator | Used in Strategy #10 |
| 26 | flat.mq5 | Indicator | Used in Strategy #10 |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4534](https://www.mql5.com/ru/articles/4534)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4534.zip "Download all attachments in the single ZIP archive")

[10Flat.zip](https://www.mql5.com/en/articles/download/4534/10flat.zip "Download 10Flat.zip")(60.96 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)
- [Quick Manual Trading Toolkit: Working with open positions and pending orders](https://www.mql5.com/en/articles/7981)
- [Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)
- [Multicurrency monitoring of trading signals (Part 5): Composite signals](https://www.mql5.com/en/articles/7759)
- [Multicurrency monitoring of trading signals (Part 4): Enhancing functionality and improving the signal search system](https://www.mql5.com/en/articles/7678)
- [Multicurrency monitoring of trading signals (Part 3): Introducing search algorithms](https://www.mql5.com/en/articles/7600)
- [Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/272700)**
(13)


![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
9 Jun 2018 at 22:00

**Alexander Fedosov:**

Thanks for the feedback. Yes, you are right, I did not accidentally use more recognisable or traditional indicators (although some of them were trending and used in the opposite way to filter or identify the signal). Before this article was about trend strategies and there I used very complex tandems or interactions of indicators with many parameters in them. Here I decided to make everything more understandable with recognisable indicators.

In my opinion, your approaches are absolutely correct. Thanks again for your interesting work.

![Vitaly Muzichenko](https://c.mql5.com/avatar/2025/11/691d3a3a-b70b.png)

**[Vitaly Muzichenko](https://www.mql5.com/en/users/mvs)**
\|
17 Jun 2018 at 21:34

**Ivan Gurov:**

Honestly, the article is rubbish. I didn't learn anything new. I'm sorry.

So you're no longer a beginner, you know everything.

![Vasily Belozerov](https://c.mql5.com/avatar/2022/10/634bb81b-1c89.png)

**[Vasily Belozerov](https://www.mql5.com/en/users/geezer)**
\|
26 Jun 2018 at 10:56

**Aleksandr Masterskikh:**

Dear Vasily!

The process of price movements of financial instruments, in general, is non-stationary. Within the framework of the impulse equilibrium theory, we have identified a structure that is constantly repeating (not just periodically, but constantly and continuously - only the parameters of this structure change). That is why it is possible to determine the period value (or frequency as an inverse value). This is absolutely impossible within the framework of traditional types of analyses (precisely because there is no mechanism for identifying such a structure).

As for your statement that "phase is a deviation from the equilibrium position" and your suggestion to use a demodulator (and even an amplitude-phase demodulator) - these are rather strange statements, especially when applied to charts of financial instruments prices (i.e. to non-stationary processes). And "in two steps", as you say, the analysis of such a complex (non-stationary) process cannot be solved. It goes like this.

1\. "a structure is identified", but "there is no mechanism to identify such a structure" - like there are aliens, but there is no mechanism to prove it.

2\. at the same time "it is possible to determine the value of the period" - we know how aliens look like, but we cannot draw them.

3\. "within the framework of traditional types of analysis it is absolutely impossible" - it is possible, but you are engaged in synthesis, not analysis.

![Martin Dieter Backschat](https://c.mql5.com/avatar/2017/10/59F0E014-BD22.jpg)

**[Martin Dieter Backschat](https://www.mql5.com/en/users/mabatrader)**
\|
20 Aug 2018 at 13:35

Hello Alexander, great article!

Can you explain about the optimization, like what time range did you use (what split In-Sample Versus Out-of-Sample; any overlappings)?

Another thought: because of the smaller than 1y [period used](https://www.mql5.com/en/docs/check/period "MQL5 documentation: Period function") here, what do think of applying a Walk-Forward optimization towards these strategies?

Thanks, Martin.

![Pit L.](https://c.mql5.com/avatar/2019/4/5CB76511-2FA6.jpg)

**[Pit L.](https://www.mql5.com/en/users/pitchief)**
\|
18 Sep 2019 at 11:33

Very good article.

Comparisons or tests of different [trading strategies](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") are especially valuable for every trader. I do not have to test everything myself and that saves me a lot of time.

I like to read such strategy comparisons, because I can analyze many results to further improve my own trading strategy.

I can put a lot of ideas for my own system on the page without having to test it myself. That saves me a lot of time and risk.

I wish there were more articles on mql that compare different trading strategies and trading systems and find out the profitability.

Very good article. Thank you Alexander for your work!

![Implementing indicator calculations into an Expert Advisor code](https://c.mql5.com/2/32/expert_indicator.png)[Implementing indicator calculations into an Expert Advisor code](https://www.mql5.com/en/articles/4602)

The reasons for moving an indicator code to an Expert Advisor may vary. How to assess the pros and cons of this approach? The article describes implementing an indicator code into an EA. Several experiments are conducted to assess the speed of the EA's operation.

![Applying the Monte Carlo method for optimizing trading strategies](https://c.mql5.com/2/32/Monte_Carlo.png)[Applying the Monte Carlo method for optimizing trading strategies](https://www.mql5.com/en/articles/4347)

Before launching a robot on a trading account, we usually test and optimize it on quotes history. However, a reasonable question arises: how can past results help us in the future? The article describes applying the Monte Carlo method to construct custom criteria for trading strategy optimization. In addition, the EA stability criteria are considered.

![How to create Requirements Specification for ordering a trading robot](https://c.mql5.com/2/32/HowCreateExpertSpecification.png)[How to create Requirements Specification for ordering a trading robot](https://www.mql5.com/en/articles/4368)

Are you trading using your own strategy? If your system rules can be formally described as software algorithms, it is better to entrust trading to an automated Expert Advisor. A robot does not need sleep or food and is not subject to human weaknesses. In this article, we show how to create Requirements Specification when ordering a trading robot in the Freelance service.

![Deep Neural Networks (Part VI). Ensemble of neural network classifiers: bagging](https://c.mql5.com/2/48/Deep_Neural_Networks_06.png)[Deep Neural Networks (Part VI). Ensemble of neural network classifiers: bagging](https://www.mql5.com/en/articles/4227)

The article discusses the methods for building and training ensembles of neural networks with bagging structure. It also determines the peculiarities of hyperparameter optimization for individual neural network classifiers that make up the ensemble. The quality of the optimized neural network obtained in the previous article of the series is compared with the quality of the created ensemble of neural networks. Possibilities of further improving the quality of the ensemble's classification are considered.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jyznkwmgsafcvjjufjerboishptwabob&ssn=1769251984284820995&ssn_dr=0&ssn_sr=0&fv_date=1769251984&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F4534&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Comparative%20analysis%20of%2010%20flat%20trading%20strategies%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925198441852907&fz_uniq=5083184277241206416&sv=2552)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).