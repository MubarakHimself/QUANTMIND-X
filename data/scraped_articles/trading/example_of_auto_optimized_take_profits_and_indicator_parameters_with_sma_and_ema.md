---
title: Example of Auto Optimized Take Profits and Indicator Parameters with SMA and EMA
url: https://www.mql5.com/en/articles/15476
categories: Trading, Machine Learning
relevance_score: 4
scraped_at: 2026-01-23T17:37:19.968362
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hzhhtvzrkxiflmwbpvrbuaepvpkqjhll&ssn=1769179038300619612&ssn_dr=0&ssn_sr=0&fv_date=1769179038&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15476&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Example%20of%20Auto%20Optimized%20Take%20Profits%20and%20Indicator%20Parameters%20with%20SMA%20and%20EMA%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691790387052231&fz_uniq=5068453776366434718&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the ever-evolving world of algorithmic trading, innovation is key to staying ahead of the curve. Today, we're excited to introduce a sophisticated Expert Advisor (EA) that combines machine learning with traditional technical analysis to navigate the forex markets. This EA, leverages an ONNX model alongside carefully optimized technical indicators to make trading decisions in the currency markets.

The EA's approach is multifaceted, utilizing price prediction from a machine learning model, trend following techniques, and adaptive parameter optimization. It's designed to operate primarily on the #AAPL stock, though it has the flexibility to be adapted for other instruments. With features like dynamic lot sizing, trailing stops, and automatic adjustment to market conditions, this EA represents a blend of cutting-edge technology and time-tested trading principles.

Explanation of Indicators Used:

1. Simple Moving Average (SMA): The EA uses a Simple Moving Average with an adaptively optimized period. The SMA helps identify the overall trend direction and is used in conjunction with price and other indicators to generate trading signals.
2. Exponential Moving Average (EMA): An Exponential Moving Average is also employed, with its period dynamically optimized. The EMA responds more quickly to recent price changes than the SMA, providing a different perspective on trend direction.
3. Average True Range (ATR): While not explicitly calculated in the code, the EA uses ATR-based calculations for setting stop-loss and take-profit levels. This allows for volatility-adjusted position sizing and risk management.
4. Machine Learning Model: The EA incorporates an ONNX (Open Neural Network Exchange) model for price prediction. This model takes in a series of recent price data and attempts to forecast the next price movement, adding a predictive element to the trading strategy.

These indicators are combined in a sophisticated manner, with their parameters being dynamically optimized based on recent market conditions. The EA also includes features like trailing stops and moral expectation calculations to manage open positions effectively.

The combination of these indicators, along with the machine learning component, allows the EA to adapt to changing market conditions and potentially identify trading opportunities across various market states.

### Breakdown of the code

1\. Initial Setup and Includes:

The code starts with copyright information and includes necessary libraries like Trade.mqh.

```
#include <Trade\Trade.mqh>
```

2\. Global Variables and Parameters:

- ONNX model parameters are defined, including sample size and handles.
- Input parameters for indicators (SMA, EMA, ATR) and trading operations are declared.
- Enums and constants for price movement and magic numbers are defined.

```
#resource "/Files/model.EURUSD.D1.1_1_2024.onnx" as uchar ExtModel[];
```

```
input group "----- Indicators Parameters -----"
int SMA_Period = 20;
int EMA_Period = 50;

input double StopLossATR = 1.5;
input double TakeProfitATR = 3.0;
input int OptimizationDays = 1;        // Hours between optimizations
input int LookbackPeriod = 7;         // Hours loockback periods
input int MinSMAPeriod = 5;            // Period min para SMA
input int MaxSMAPeriod = 50;           // Periodo max para SMA
input int MinEMAPeriod = 5;            // Periodo min para EMA
input int MaxEMAPeriod = 50;           // Periodo max para EMA
```

```
#define MAGIC_SE 12321
```

```
datetime lastOptimizationTime = 0;
double optimizedTakeProfit = 0.0;//InpTakeProfit;
double optimizedStopLoss = 0.0;//InpStopLoss;
```

```
double InpTakeProfit1 ;
double InpStopLoss1;
```

3\. Initialization Function (OnInit):

- Sets up the ONNX model from a buffer.
- Initializes technical indicators (SMA, EMA).
- Calls functions to optimize indicators and trading parameters.

```
int OnInit()
  {
//--- create a model from static buffer
   ExtHandle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(ExtHandle == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ", GetLastError());
      return(INIT_FAILED);
     }

//--- set input and output shapes
   const long input_shape[] = {1, SAMPLE_SIZE, 1};
   if(!OnnxSetInputShape(ExtHandle, ONNX_DEFAULT, input_shape))
     {
      Print("OnnxSetInputShape error ", GetLastError());
      return(INIT_FAILED);
     }
   const long output_shape[] = {1, 1};
   if(!OnnxSetOutputShape(ExtHandle, 0, output_shape))
     {
      Print("OnnxSetOutputShape error ", GetLastError());
      return(INIT_FAILED);
     }

   SMAHandle = iMA(_Symbol, _Period, SMA_Period, 0, MODE_SMA, PRICE_CLOSE); // Ensure correct period
   if(SMAHandle == INVALID_HANDLE)
     {
      Print("Error initializing SMA indicator: ", GetLastError());
      return INIT_FAILED;
     }
   EMAHandle = iMA(_Symbol, _Period, EMA_Period, 0, MODE_EMA, PRICE_CLOSE); // Ensure correct index
   if(EMAHandle == INVALID_HANDLE)
     {
      Print("Error initializing EMA indicator: ", GetLastError());
      return INIT_FAILED;
     }
```

```
   trade.SetDeviationInPoints(Slippage);

   trade.SetExpertMagicNumber(MAGIC_SE);
```

```
   OptimizeIndicators();
   OptimizeParameters();
   return(INIT_SUCCEEDED);
  }
```

4\. Deinitialization Function (OnDeinit):

Releases handles for the ONNX model and indicators.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(ExtHandle != INVALID_HANDLE)
     {
      OnnxRelease(ExtHandle);
      ExtHandle = INVALID_HANDLE;
     }

   IndicatorRelease(SMAHandle);
   IndicatorRelease(EMAHandle);
  }
```

5\. Main Trading Logic (OnTick):

- Checks if the market is closed.
- Periodically optimizes indicators and trading parameters.
- Updates trailing stop logic.
- Predicts price movement using the ONNX model.
- Checks for open/close position conditions based on predictions and indicators.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   if(IsMarketClosed())  // Verificar si el mercado está cerrado
     {
      return; // Si el mercado está cerrado, no hacer nada
     }

   static datetime lastOptimizationTime2 = 0;

   if(TimeCurrent() - lastOptimizationTime2 >= OptimizationDays * PeriodSeconds(PERIOD_H1))
     {
      OptimizeIndicators();
      lastOptimizationTime2 = TimeCurrent();

      // Actualizar los indicadores con los nuevos períodos
      IndicatorRelease(SMAHandle);
      IndicatorRelease(EMAHandle);
      SMAHandle = iMA(_Symbol, _Period, SMA_Period, 0, MODE_SMA, PRICE_CLOSE);
      EMAHandle = iMA(_Symbol, _Period, EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
     }

//--- Optimización cada 2 días
   if(TimeCurrent() - lastOptimizationTime >= PeriodSeconds(PERIOD_H1) * HoursAnalyze)
     {
      OptimizeParameters();
      lastOptimizationTime = TimeCurrent();
     }

//---

   if(NewBarTS()==true)//gather statistics and launch trailing stop
     {
      double open=iOpen(_Symbol,TFTS,1);
      CalcLvl(up,(int)MathRound((iHigh(_Symbol,TFTS,1)-open)/_Point));
      CalcLvl(dn,(int)MathRound((open-iLow(_Symbol,TFTS,1))/_Point));
      buy_sl=CalcSL(dn);
      buy_tp=CalcTP(up);
      sell_sl=CalcSL(up);
      sell_tp=CalcTP(dn);

      if(TypeTS==Simple)//simple trailing stop
         SimpleTS();

      if(TypeTS==MoralExp)//Moral expectation
         METS();
      if(TypeTS==None)//None TS
         return;
     }

   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);

   if(bid==SLNeutral || bid<=SLBuy || (SLSell>0 && bid>=SLSell))
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong ticket=PositionGetTicket(i);
         if(PositionSelectByTicket(ticket)==true)
            trade.PositionClose(ticket);
        }
     }
//---
//--- check new day
   if(TimeCurrent() >= ExtNextDay)
     {
      GetMinMax();
      ExtNextDay = TimeCurrent();
      ExtNextDay -= ExtNextDay % PeriodSeconds(PERIOD_D1);
      ExtNextDay += PeriodSeconds(PERIOD_D1);
     }

//--- check new bar
   if(TimeCurrent() < ExtNextBar)
      return;
   ExtNextBar = TimeCurrent();
   ExtNextBar -= ExtNextBar % PeriodSeconds();
   ExtNextBar += PeriodSeconds();

//--- check min and max
   float close = (float)iClose(_Symbol, _Period, 0);
   if(ExtMin > close)
      ExtMin = close;
   if(ExtMax < close)
      ExtMax = close;

   double sma[], ema[];//, willr[];
   CopyBuffer(SMAHandle, 0, 0, 1, sma);
   CopyBuffer(EMAHandle, 0, 0, 1, ema);
//CopyBuffer(WillRHandle, 0, 0, 1, willr);

//--- predict next price
   PredictPrice();

//--- check trading according to prediction and indicators
   if(ExtPredictedClass >= 0)
     {
      if(PositionSelect(_Symbol))
         CheckForClose(sma[0], ema[0]);//, willr[0]);
      else
         CheckForOpen(sma[0], ema[0]);//, willr[0]);
     }
  }
```

6\. Trading Functions:

- CheckForOpen: Determines whether to open a buy or sell position based on predictions and indicator signals.
- CheckForClose: Checks if current positions should be closed based on predictions.

```
//+------------------------------------------------------------------+
//| Check for open position conditions                               |
//+------------------------------------------------------------------+
void CheckForOpen(double sma, double ema)//, double willr)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int copied = CopyRates(_Symbol,0,0,1,rates);
   if(copied <= 0)
     {
      Print("Error copying rates: ", GetLastError());
      return;
     }
   double Close[1];
   Close[0]=rates[0].close;
   double close = Close[0];

   ENUM_ORDER_TYPE signal = WRONG_VALUE;
   Print("ExtPredictedClass ",ExtPredictedClass);

//--- check signals
   if(ExtPredictedClass == 2)//PRICE_DOWN)
     {
      Print("ExtPredictedClass Sell ",ExtPredictedClass);
      Print("close ",close, " sma ",sma, " ema ", ema);//, " willr ", willr);
      // Venta
      if((close < sma && close < ema))// || willr > -20)
        {
         signal = ORDER_TYPE_SELL;
         Print("Order Sell detected");
        }
     }
   else
      if(ExtPredictedClass == 0)//PRICE_UP)
        {
         Print("ExtPredictedClass Buy ",ExtPredictedClass);
         Print("close ",close, " sma ",sma, " ema ", ema);//, " willr ", willr);
         // Compra
         if((close > sma && close > ema))// || willr < -80)
           {
            signal = ORDER_TYPE_BUY;
            Print("Order Buy detected");
           }
        }

//--- open position if possible according to signal
   if(signal != WRONG_VALUE && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
     {
      Print("Proceding open order");
      double price, sl=0, tp=0;
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      MqlTradeRequest request = {};
      MqlTradeResult result = {};

      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.deviation = Slippage;
      request.magic = MAGIC_SE;
      request.type_filling = ORDER_FILLING_FOK;
      //request.comment = "AKWr";

      double lotaje;
      if(signal == ORDER_TYPE_SELL)
        {
         price = bid;
         Print("Price: ",price);
         if(inp_lot_type == LOT_TYPE_FIX)
            lotaje=inp_lot_fix ;
         else
            lotaje=get_lot(price);
         if(!CheckVolumeValue(lotaje))
            return;
         if(!InpUseStops && ATR)
           {
            sl = NormalizeDouble(bid + StopLossATR * ATRValue, _Digits);
            tp = NormalizeDouble(ask - TakeProfitATR * ATRValue, _Digits);
            if(!CheckMoneyForTrade(_Symbol, lotaje,ORDER_TYPE_SELL))
              {
               Print("No hay suficiente margen para abrir la posición");
               return;
              }
            request.type = ORDER_TYPE_SELL;
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            request.volume = lotaje;
            request.sl = sl;
            request.tp = tp;
            request.comment = "SEW Opened sell order";
           }
         if(!InpUseStops && ATR)
           {
            sl = 0;
            tp = 0;
           }
         else
           {
            InpTakeProfit1 =optimizedTakeProfit;
            InpStopLoss1= optimizedStopLoss;
            sl = NormalizeDouble(bid + InpStopLoss1*_Point, _Digits);
            tp = NormalizeDouble(ask - InpTakeProfit1*_Point, _Digits);

           }
        }
      else
        {
         price = ask;
         Print("Price: ",price);
         if(inp_lot_type == LOT_TYPE_FIX)
            lotaje=inp_lot_fix ;
         else
            lotaje=get_lot(price);
         if(!CheckVolumeValue(lotaje))
            return;
         if(!InpUseStops)
           {
            sl = NormalizeDouble(ask - StopLossATR * ATRValue, _Digits);
            tp = NormalizeDouble(bid + TakeProfitATR * ATRValue, _Digits);
            if(!CheckMoneyForTrade(_Symbol, lotaje,ORDER_TYPE_BUY))
              {
               Print("No hay suficiente margen para abrir la posición");
               return;
              }
            request.type = ORDER_TYPE_BUY;
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            request.volume = lotaje;
            request.sl = sl;
            request.tp = tp;
            request.comment = "SEW Opened buy order";
           }
         if(!InpUseStops && ATR)
           {
            sl = 0;
            tp = 0;
           }
         else
           {
            InpTakeProfit1 =optimizedTakeProfit;
            InpStopLoss1= optimizedStopLoss;
            sl = NormalizeDouble(ask - InpStopLoss1*_Point, _Digits);
            tp = NormalizeDouble(bid + InpTakeProfit1*_Point, _Digits);
           }
        }
      Print("No InpUseStops used");

      //ExtTrade.PositionOpen(_Symbol, signal, lotaje, price, sl, tp);

      if(!CheckMoneyForTrade(_Symbol, lotaje, (ENUM_ORDER_TYPE)signal))
        {
         Print("No hay suficiente margen para abrir la posición");
         return;
        }
      Print("Volume ", lotaje);
      request.type = signal;
      request.price = price;//SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.volume = lotaje;
      request.sl = sl;
      request.tp = tp;
      request.comment = "SEW";
      if(!OrderSend(request, result))
        {
         Print("Error opening the order: ", GetLastError());
         return;
        }
     }
  }
//+------------------------------------------------------------------+
//| Check for close position conditions                              |
//+------------------------------------------------------------------+
void CheckForClose(double sma, double ema)//, double willr)
  {
   if(InpUseStops)
      return;

   bool bsignal = false;

//--- position already selected before
   long type = PositionGetInteger(POSITION_TYPE);

//--- check signals
   if(type == POSITION_TYPE_BUY && ExtPredictedClass == 2)//PRICE_DOWN)
      bsignal = true;
   if(type == POSITION_TYPE_SELL && ExtPredictedClass == 0)//PRICE_UP)
      bsignal = true;

//--- close position if possible
   if(bsignal && TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
     {
      ExtTrade.PositionClose(_Symbol);
      CheckForOpen(sma, ema);//, willr);
     }
  }
```

7\. Price Prediction (PredictPrice):

Uses the ONNX model to predict future price movement.

```
//+------------------------------------------------------------------+
//| Predict next price                                               |
//+------------------------------------------------------------------+
void PredictPrice(void)
  {
   static vectorf output_data(1);
   static vectorf x_norm(SAMPLE_SIZE);

   if(ExtMin >= ExtMax)
     {
      Print("ExtMin >= ExtMax");
      ExtPredictedClass = -1;
      return;
     }

   if(!x_norm.CopyRates(_Symbol, _Period, COPY_RATES_CLOSE, 1, SAMPLE_SIZE))
     {
      Print("CopyRates ", x_norm.Size());
      ExtPredictedClass = -1;
      return;
     }
   float last_close = x_norm[SAMPLE_SIZE - 1];
   x_norm -= ExtMin;
   x_norm /= (ExtMax - ExtMin);

   if(!OnnxRun(ExtHandle, ONNX_NO_CONVERSION, x_norm, output_data))
     {
      Print("OnnxRun");
      ExtPredictedClass = -1;
      return;
     }

   float predicted = output_data[0] * (ExtMax - ExtMin) + ExtMin;
   float delta = last_close - predicted;
   if(fabs(delta) <= 0.00001)
      ExtPredictedClass = PRICE_SAME;
   else
      if(delta < 0)
         ExtPredictedClass = PRICE_UP;
      else
         ExtPredictedClass = PRICE_DOWN;

// Debugging output
   Print("Predicted price: ", predicted, " Delta: ", delta, " Predicted Class: ", ExtPredictedClass);
  }
```

8\. Trailing Stop Functions:

Various functions (AllTS, METS, SimpleTS) implement different trailing stop strategies.

All explained in this [article](https://www.mql5.com/en/articles/14167):  [Trailing stop in trading - MQL5 Articles](https://www.mql5.com/en/articles/14167)

9\. Optimization Functions:

- OptimizeParameters: Tries different take profit and stop loss values to find optimal settings.
- OptimizeIndicators: Finds the best periods for SMA and EMA indicators.

```
void OptimizeParameters()
  {
   double bestTakeProfit = InpTakeProfit1;
   double bestStopLoss = InpStopLoss1;
   double bestPerformance = -DBL_MAX;

   for(int tp = 65; tp <= 500; tp += 5) // rango de TakeProfit
     {
      for(int sl = 65; sl <= 500; sl += 5) // rango de StopLoss
        {
         double performance = TestStrategy(tp, sl);
         if(performance > bestPerformance)
           {
            bestPerformance = performance;
            bestTakeProfit = tp;
            bestStopLoss = sl;
            //Print("Best Take Profit",bestTakeProfit);
            //Print("Best Stop Loss",bestStopLoss);
           }
        }
     }

   optimizedTakeProfit = bestTakeProfit;
   optimizedStopLoss = bestStopLoss;

   Print("Optimized TakeProfit: ", optimizedTakeProfit);
   Print("Optimized StopLoss: ", optimizedStopLoss);
  }
```

```
void OptimizeIndicators()
  {
   datetime startTime = TimeCurrent() - LookbackPeriod * PeriodSeconds(PERIOD_H1);
   datetime endTime = TimeCurrent();

   int bestSMAPeriod = SMA_Period;
   int bestEMAPeriod = EMA_Period;
   double bestPerformance = -DBL_MAX;

   for(int smaPeriod = MinSMAPeriod; smaPeriod <= MaxSMAPeriod; smaPeriod++)
     {
      for(int emaPeriod = MinEMAPeriod; emaPeriod <= MaxEMAPeriod; emaPeriod++)
        {
         double performance = TestIndicatorPerformance(smaPeriod, emaPeriod, startTime, endTime);

         if(performance > bestPerformance)
           {
            bestPerformance = performance;
            bestSMAPeriod = smaPeriod;
            bestEMAPeriod = emaPeriod;
           }
        }
     }

   SMA_Period = bestSMAPeriod;
   EMA_Period = bestEMAPeriod;

   Print("Optimized SMA Period: ", SMA_Period);
   Print("Optimized EMA Period: ", EMA_Period);
  }
```

10\. Utility Functions & Money Management: :

Functions for lot size calculation, volume checking, market closure checking,  includes functions to check if there's enough money for trades and to normalize lot sizes, etc.

```
bool IsMarketClosed()
  {
   datetime currentTime = TimeCurrent();
   MqlDateTime tm;
   TimeToStruct(currentTime, tm);

   int dayOfWeek = tm.day_of_week;
   int hour = tm.hour;

// Verifica si es fin de semana
   if(dayOfWeek <= Sunday || dayOfWeek >= Saturday)
     {
      return true;
     }

// Verifica si está fuera del horario habitual de mercado (ejemplo: 21:00 a 21:59 UTC)
   if(hour >= after || hour < before)  // Ajusta estos valores según el horario del mercado
     {
      return true;
     }

   return false;
  }

//+------------------------------------------------------------------+
//| Check the correctness of the order volume                        |
//+------------------------------------------------------------------+
bool CheckVolumeValue(double volume)//,string &description)
  {
//--- minimal allowed volume for trade operations
   double min_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   if(volume<min_volume)
     {
      //description=StringFormat("Volume is less than the minimal allowed SYMBOL_VOLUME_MIN=%.2f",min_volume);
      return(false);
     }

//--- maximal allowed volume of trade operations
   double max_volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MAX);
   if(volume>max_volume)
     {
      //description=StringFormat("Volume is greater than the maximal allowed SYMBOL_VOLUME_MAX=%.2f",max_volume);
      return(false);
     }

//--- get minimal step of volume changing
   double volume_step=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_STEP);

   int ratio=(int)MathRound(volume/volume_step);
   if(MathAbs(ratio*volume_step-volume)>0.0000001)
     {
      //description=StringFormat("Volume is not a multiple of the minimal step SYMBOL_VOLUME_STEP=%.2f, the closest correct volume is %.2f",
      //volume_step,ratio*volume_step);
      return(false);
     }
//description="Correct volume value";
   return(true);
  }
//+------------------------------------------------------------------+
bool CheckMoneyForTrade(string symb,double lots,ENUM_ORDER_TYPE type)
  {
//--- Getting the opening price
   MqlTick mqltick;
   SymbolInfoTick(symb,mqltick);
   double price=mqltick.ask;
   if(type==ORDER_TYPE_SELL)
      price=mqltick.bid;
//--- values of the required and free margin
   double margin,free_margin=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
//--- call of the checking function
   if(!OrderCalcMargin(type,symb,lots,price,margin))
     {
      //--- something went wrong, report and return false
      Print("Error in ",__FUNCTION__," code=",GetLastError());
      return(false);
     }
//--- if there are insufficient funds to perform the operation
   if(margin>free_margin)
     {
      //--- report the error and return false
      Print("Not enough money for ",EnumToString(type)," ",lots," ",symb," Error code=",GetLastError());
      return(false);
     }
//--- checking successful
   return(true);
  }
```

```
double get_lot(double price)
  {
   if(inp_lot_type==LOT_TYPE_FIX)
      return(normalize_lot(inp_lot_fix));
   double one_lot_margin;
   if(!OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,1.0,price,one_lot_margin))
      return(inp_lot_fix);
   return(normalize_lot((AccountInfoDouble(ACCOUNT_BALANCE)*(inp_lot_risk/100))/ one_lot_margin));
  }
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
double normalize_lot(double lt)
  {
   double lot_step = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   lt = MathFloor(lt / lot_step) * lot_step;
   double lot_minimum = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   lt = MathMax(lt, lot_minimum);
   return(lt);
  }
```

Key Features:

1. Uses machine learning (ONNX model) for price prediction.
2. Combines technical indicators (SMA, EMA) with ML predictions for trading decisions.
3. Implements multiple trailing stop strategies.
4. Includes periodic optimization of indicator parameters and trading settings.
5. Has built-in risk management (lot sizing, money checking).
6. Considers market hours for trading.

This Expert Advisor (EA) is quite sophisticated, combining traditional technical analysis with machine learning for forex trading. It also includes various risk management and optimization features to adapt to changing market conditions.

![Inputs](https://c.mql5.com/2/87/inputs.png)

![Graph AAPL](https://c.mql5.com/2/87/TesterGraphReport2024.08.01.png)

![Backtesting AAPL](https://c.mql5.com/2/87/backtesting.png)

This analysis indicates that the automated trading strategy demonstrated profitability with a solid Sharpe Ratio of 6.21. However, it has a high drawdown that sugest the need for careful risk management. The strategy's ability to achieve consistent gains, as shown by the equity curve, reflects its potential for real-world trading applications. Future optimizations could focus on reducing drawdowns and improving the recovery factor to enhance overall performance.

### Summary

This article introduces an innovative Expert Advisor (EA) for algorithmic trading in the forex market, specifically designed for trading Apple Inc. (#AAPL) stock. The EA represents a sophisticated blend of machine learning and traditional technical analysis, aiming to navigate the complexities of currency markets with precision and adaptability.

At the core of this EA is an ONNX (Open Neural Network Exchange) model, which serves as the machine learning component. This model is tasked with predicting price movements based on recent market data, adding a forward-looking element to the trading strategy. The EA combines these predictions with established technical indicators, namely Simple Moving Average (SMA) and Exponential Moving Average (EMA), to generate trading signals.

The EA's approach is multifaceted, incorporating several key features:

1. Dynamic Optimization: Both the technical indicators and trading parameters are subject to periodic optimization. This allows the EA to adapt to changing market conditions, potentially improving its performance over time.
2. Adaptive Risk Management: The EA employs dynamic lot sizing and uses Average True Range (ATR) calculations for setting stop-loss and take-profit levels. This approach aims to adjust position sizes and risk exposure based on current market volatility.
3. Multiple Trailing Stop Strategies: The EA implements various trailing stop methods, allowing for flexible management of open positions and potentially maximizing profits while minimizing losses.
4. Market Condition Awareness: The system is designed to consider market hours and conditions, ensuring that trades are only executed when appropriate.

The article provides a detailed breakdown of the EA's code structure, elucidating its key components:

1. Initialization: This phase sets up the ONNX model and technical indicators, preparing the EA for operation.
2. Main Trading Logic: The core functionality that decides when to open or close positions based on the combined signals from the machine learning model and technical indicators.
3. Price Prediction: Utilizes the ONNX model to forecast future price movements.
4. Optimization Functions: Periodically adjusts indicator parameters and trading settings to maintain effectiveness in varying market conditions.
5. Risk Management: Includes functions for lot size calculation, money management, and market condition checking.

The EA's performance was evaluated through backtesting on AAPL stock data. The results showed promising profitability with a Sharpe Ratio of 6.21, indicating strong risk-adjusted returns. However, the analysis also revealed relatively high drawdown figures, suggesting areas for potential improvement in risk management.

### Conclusion

Well the Expert Advisor (EA) we have today is a huge leap. It uses a mix of smart technology (machine learning) and proven methods(technical analysis) to provide traders with invaluable members-only decisions on how investing in AAPL stocks and other symbols.

The EA combined an ONNX model to predict prices with optimized technical indicators. This blend implies that it can respond to both instant price changes and slow-burner movements into longer-term developments.

One of the appealing things about this EA is that it has a cohesive risk management system. For example, it employs dynamic lot sizing (change your trade sizes according to market conditions), ATR-based stop-loss and take-profit levels (impose limits on what you stand to lose should things go sour or winnings when they are in your favour) as well as multiple trailing stop strategies with respect of price change. These tools are tailored to keep your money without being cheesed out in undisclosed ways.

Another great thing about this EA is that it updates its indicators and trading parameters quite regularly. This will enable it to keep pace with the changing market, a very important component to remain effective over time.

The results, however, are still somewhat disappointing from the backtesting analysis that shows how the EA would have performed in the past. Although the results seem good in terms of profits and risk management, it did show some high drawdowns—a big drop in value—thereby indicating that the risk management could be tightened up. Future versions of the EA may include better safeguards or more cautious trading during very volatile times.

Although it is created to trade AAPL stock, the principles working in it may be easily applied for other financial markets. Such flexibility makes this instrument worth using both in present trading and as a model for future algorithmic systems.

This is a complex, highly promising EA of automated trading—a blend of cutting-edge tech and traditional ways, strong in risk management and adaptable to changing market conditions. However, being like any other trading tool, it does require regular monitoring, updating, and testing in real-world scenarios to ensure its smooth functionality over time.

Finally, while this EA seems to have a lot of potential but more work on it has to be made, and lots of optimizing have to be done, it’s important to remember that trading always involves risk, and past performance doesn’t guarantee future success. If you’re considering using this or any automated system, make sure you understand the risks, do your homework, and ideally test it in a simulated environment before using real money.

I hope you enjoy reading this article as I enjoy writing it, I hope you can make this EA much better and obtain good results. This is a good example of how to implement auto optimization with stops and with indicators. Once again, I hope you liked this. Cheers!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15476.zip "Download all attachments in the single ZIP archive")

[SE\_v9.mq5](https://www.mql5.com/en/articles/download/15476/se_v9.mq5 "Download SE_v9.mq5")(80.02 KB)

[model.6AAPL.D1.1\_1\_2024.onnx](https://www.mql5.com/en/articles/download/15476/model.6aapl.d1.1_1_2024.onnx "Download model.6AAPL.D1.1_1_2024.onnx")(884.45 KB)

[model.EURUSD.D1.1\_1\_2024.onnx](https://www.mql5.com/en/articles/download/15476/model.eurusd.d1.1_1_2024.onnx "Download model.EURUSD.D1.1_1_2024.onnx")(884.4 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications](https://www.mql5.com/en/articles/16682)
- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/471232)**
(5)


![Arup Nag](https://c.mql5.com/avatar/avatar_na2.png)

**[Arup Nag](https://www.mql5.com/en/users/arupnag)**
\|
16 Aug 2024 at 09:12

A step by step guide for [auto optimization](https://www.mql5.com/en/articles/7538 "Article: Continuous Sliding Optimization (Part 4): Program for Optimization Management (Auto-Optimizer) ") will be very helpful


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
10 Sep 2024 at 11:43

**Arup Nag [#](https://www.mql5.com/en/forum/471232#comment_54314212):**

A step by step guide for [auto optimization](https://www.mql5.com/en/articles/7538 "Article: Continuous Sliding Optimization (Part 4): Program for Optimization Management (Auto-Optimizer) ") will be very helpful

Thanks! I am making one just now, it would be finished in not much time.Please ask for anything else you want or need.

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
13 Sep 2024 at 21:19

**Arup Nag [#](https://www.mql5.com/en/forum/471232#comment_54314212):**

A step by step guide for [auto optimization](https://www.mql5.com/en/articles/7538 "Article: Continuous Sliding Optimization (Part 4): Program for Optimization Management (Auto-Optimizer) ") will be very helpful

Here is the article, I hope you like it:  [How to Implement Auto Optimization in MQL5 Expert Advisors - MQL5 Articles](https://www.mql5.com/en/articles/15837)

![yehaichang](https://c.mql5.com/avatar/2025/4/67f6d33b-abb3.jpg)

**[yehaichang](https://www.mql5.com/en/users/yehaichang)**
\|
9 Apr 2025 at 20:00

Is there any EA based on [neural network trading](https://www.mql5.com/en/articles/7370 "Article: Practical application of neural networks in trading. Let's go to practice"), I would like to try, by adjusting the parameters of the high winning rate, I am going to buy!


![qademir](https://c.mql5.com/avatar/avatar_na2.png)

**[qademir](https://www.mql5.com/en/users/qademir)**
\|
4 Oct 2025 at 11:47

As I live in Brazil and would like to trade the mini index, how would I go about adapting the onnx?

adapt the onnx? files for the B3 mini index?

Thank you very much

_< edited by moderator_

Ademir J Dias

It is forbidden to post personal details such as telephone [numbers](https://www.mql5.com/pt/forum/339131#:~:text=%C3%89%20proibido%20postar%20dados%20pessoais%2C%20como%20n%C3%BAmeros%20de%20telefone%20ou%20endere%C3%A7os%20de%20e%2Dmail.) or e-mail addresses.

![Developing a multi-currency Expert Advisor (Part 6): Automating the selection of an instance group](https://c.mql5.com/2/74/Developing_a_multi-currency_advisor_Part_1___LOGO__4.png)[Developing a multi-currency Expert Advisor (Part 6): Automating the selection of an instance group](https://www.mql5.com/en/articles/14478)

After optimizing the trading strategy, we receive sets of parameters. We can use them to create several instances of trading strategies combined in one EA. Previously, we did this manually. Here we will try to automate this process.

![MQL5 Wizard Techniques you should know (Part 31): Selecting the Loss Function](https://c.mql5.com/2/88/MQL5_Wizard_Techniques_you_should_know_Part_31___LOGO4.png)[MQL5 Wizard Techniques you should know (Part 31): Selecting the Loss Function](https://www.mql5.com/en/articles/15524)

Loss Function is the key metric of machine learning algorithms that provides feedback to the training process by quantifying how well a given set of parameters are performing when compared to their intended target. We explore the various formats of this function in an MQL5 custom wizard class.

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 2): Sending Signals from MQL5 to Telegram](https://c.mql5.com/2/88/logo-Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_sPart_1u.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 2): Sending Signals from MQL5 to Telegram](https://www.mql5.com/en/articles/15495)

In this article, we create an MQL5-Telegram integrated Expert Advisor that sends moving average crossover signals to Telegram. We detail the process of generating trading signals from moving average crossovers, implementing the necessary code in MQL5, and ensuring the integration works seamlessly. The result is a system that provides real-time trading alerts directly to your Telegram group chat.

![Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(I)-Fine-tuning](https://c.mql5.com/2/88/Tuning_LLMs_with_Your_Own_Personalized_Data_and_Integrating_into_EA_Part_5__LOGO.png)[Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(I)-Fine-tuning](https://www.mql5.com/en/articles/13497)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/15476&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068453776366434718)

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